# Copyright (C) 2020 Alibaba Group Holding Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#==========================================================================

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import array
import json
import logging
import os
import sys
import threading
import time
import math
from queue import Queue
from collections import OrderedDict

import mlperf_loadgen as lg
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import glob
from torch2trt import torch2trt, TRTModule
import tensorrt as trt

logging.basicConfig(
    format='%(asctime)s [%(name)s] %(message)s', level=logging.INFO)
log = logging.getLogger("main")
assert (torch.cuda.is_available()), "gpu is not available!"

NANO_SEC = 1e9

SCENARIO_MAP = {
    "SingleStream": lg.TestScenario.SingleStream,
    "MultiStream": lg.TestScenario.MultiStream,
    "Server": lg.TestScenario.Server,
    "Offline": lg.TestScenario.Offline,
}

last_timeing = []


def get_args():
  """Parse commandline."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--dataset-path", required=True, help="path to the dataset")
  parser.add_argument(
      "--model-builder-dir",
      required=True,
      help="the directory of once-for-all")
  parser.add_argument(
      "--scenario",
      default="Offline",
      choices=['Offline'],
      help="mlperf benchmark scenario")
  parser.add_argument(
      "--batch-size", type=int, help="batch size in a single inference")
  parser.add_argument("--model", help="model name")
  parser.add_argument("--output", default="output", help="test results")
  parser.add_argument("--threads", default=2, type=int, help="threads")
  parser.add_argument("--cache", type=int, default=1, help="use cache")
  parser.add_argument(
      "--accuracy", action="store_true", help="enable accuracy pass")
  parser.add_argument(
      "--config", default="mlperf.conf", help="mlperf rules config")
  parser.add_argument("--user-config", help="user config")
  parser.add_argument("--image_size", type=int, default=172, help="image size")
  parser.add_argument(
      "--audit-test",
      type=str,
      choices=['1', '4a', '4b', '5'],
      help="audit test mode")
  parser.add_argument(
      "--force-build", action="store_true", help="force build tensorrt engine")
  parser.add_argument(
      '--calib-file',
      type=int,
      choices=[1, 2],
      default=2,
      help='calib file id')
  parser.add_argument(
      "--calib-algo",
      type=int,
      default=1,
      choices=[1, 2, 3],
      help="calibration algorithm")
  parser.add_argument(
      "--calib-batch-size",
      type=int,
      default=100,
      help="calibration batch size")
  args = parser.parse_args()

  args.chip_name = torch.cuda.get_device_name(0).split()[-1].split('-')[0]
  
  if args.model is None:
    args.model = 'best_model'
  
  if args.user_config is None:
    args.user_config = 'user_' + args.chip_name + '.conf'
    
  if args.batch_size is None:
    args.batch_size = 1250 if args.accuracy else 1280

  return args


class BackendTensorRT:
  def __init__(self):
    self.model = None

  def version(self):
    return torch.__version__

  def name(self):
    return "pytorch-tensorrt-ofa"

  def load(self, args, ds=None):
    prefix = 'bs%d_is%d_%s_' % (args.batch_size, args.image_size, args.chip_name)
    lib_name = 'pretrained/' + args.model + '/' + prefix + 'torch2trt.pth'
    if os.path.exists(lib_name) and not args.force_build:
      self.model = TRTModule()
      self.model.load_state_dict(torch.load(lib_name))
      self.model.eval()
    else:
      net, _ = load_model(args)
      net = net.cuda()
      net.eval()
      input_data = torch.FloatTensor(
          np.array(ds.get_calibration_set(), np.float32)).cuda()
      if args.calib_algo == 1:
        calib_algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION
      elif args.calib_algo == 2:
        calib_algo = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
      elif args.calib_algo == 3:
        calib_algo = trt.CalibrationAlgoType.MINMAX_CALIBRATION
      size = 1 << (33 if 'T4' in args.chip_name else 34)
      self.model = torch2trt(
          net, [input_data],
          max_batch_size=args.batch_size,
          fp16_mode=True,
          max_workspace_size=size,
          int8_mode=True,
          int8_calib_algorithm=calib_algo,
          int8_calib_batch_size=args.calib_batch_size)
      torch.save(self.model.state_dict(), lib_name)
    log.info('model is ready')
    return self

  def predict(self, image):
    with torch.no_grad():
      output = self.model(image)
      _, output = output.max(1)
    return output


class ImageFolderWithPaths(datasets.ImageFolder):
  def __getitem__(self, index):
    original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
    path = self.imgs[index][0]
    tuple_with_path = (original_tuple + (path, ))
    return tuple_with_path


class Imagenet:
  def __init__(self,
               data_path,
               use_cache=0,
               image_size=None,
               cache_dir=None,
               batch_size=32,
               calib_file=None):
    self.VAL_SET_SIZE = 50000
    self.last_loaded = -1
    self.image_size = image_size if image_size is not None else 224
    if not cache_dir:
      cache_dir = os.getcwd()
    self.cache_dir = os.path.join(cache_dir, "preprocessed",
                                  "imagenet_%d" % image_size)
    os.makedirs(self.cache_dir, exist_ok=True)
    self.use_cache = use_cache
    self.calib_file = calib_file
    self.data_path = data_path
    self.batch_size = batch_size
    self.data = None
    self.labels = None

    self.preprocess_image()

  def preprocess_image(self):
    self.image_list = []
    self.label_list = []
    start = time.time()
    cached = glob.glob(self.cache_dir + '/*npy')
    imageName2label = {}
    imageName2Data = {}
    if self.use_cache == 1 and len(cached) == self.VAL_SET_SIZE:
      for image_name in cached:
        image_name = os.path.splitext(os.path.basename(image_name))[0]
        label = int(os.path.splitext(image_name)[1][1:])
        imageName2label[image_name] = label
    else:
      data_loader = torch.utils.data.DataLoader(
          ImageFolderWithPaths(
              self.data_path,
              transforms.Compose([
                  transforms.Resize(int(math.ceil(self.image_size / 0.875))),
                  transforms.CenterCrop(self.image_size),
                  transforms.ToTensor(),
                  transforms.Normalize(
                      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
              ])),
          batch_size=self.batch_size,
          num_workers=10,
          pin_memory=True,
      )
      for images, labels, paths in data_loader:
        for image, label, path in zip(images, labels, paths):
          image_name = os.path.splitext(
              os.path.basename(path))[0] + ".%d" % int(label)
          if self.use_cache == 0:
            imageName2Data[image_name] = image
          else:
            dst = os.path.join(self.cache_dir, image_name)
            if not os.path.exists(dst + ".npy"):
              np.save(dst, image.detach().cpu().numpy())
          imageName2label[image_name] = int(label)

    for image_name in sorted(imageName2label.keys()):
      if self.use_cache == 1:
        self.image_list.append(image_name)
      else:
        self.image_list.append(imageName2Data[image_name])
      self.label_list.append(imageName2label[image_name])

    if not self.image_list:
      log.error("no images in image list found")
      raise ValueError("no images in image list found")

    not_found = self.VAL_SET_SIZE - len(self.image_list)
    if not_found > 0:
      log.info("reduced image list, %d images not found", not_found)
    time_taken = time.time() - start
    log.info("loaded {} images, cache={}, took={:.1f}sec".format(
        len(self.image_list), self.use_cache, time_taken))
    self.label_list = np.array(self.label_list)

  def get_calibration_set(self):
    data = []
    sample_list = []
    with open(self.calib_file, 'r') as fi:
      for line in fi:
        sample_list.append(int(line[15:23]))
    for imageId in sample_list:
      image, _ = self.get_item(imageId)
      data.append(image)
    return data

  def load_query_samples(self, sample_list):
    start = time.time()
    self.cpuData = []
    labels = []
    for imageId in sample_list:
      image, label = self.get_item(imageId)
      self.cpuData.append(image)
      labels.append(label)
    nparray = np.array(self.cpuData, np.float32)
    self.cpuData = torch.FloatTensor(nparray)
    self.labels = labels
    self.last_loaded = time.time()
    log.info('loaded %d samples in %.5f s' % (len(sample_list),
                                              time.time() - start))

  def get_samples(self):
    if self.cpuData is not None:
      self.data = self.cpuData.cuda()
      self.cpuData = None
    return self.data, self.labels

  def unload_query_samples(self, sample_list):
    self.data = None
    self.labels = None

  def get_item(self, nr):
    """Get image by number in the list."""
    if self.use_cache == 1:
      dst = os.path.join(self.cache_dir, self.image_list[nr])
      img = np.load(dst + ".npy")
    else:
      img = self.image_list[nr].detach().cpu().numpy()
    return img, self.label_list[nr]

  def get_item_count(self):
    return len(self.image_list)


class PostProcessCommon:
  def __init__(self, offset=0):
    self.offset = offset
    self.good = 0
    self.total = 0

  def __call__(self, results, expected=None, result_dict=None):
    processed_results = []
    for result, expect in zip(results, expected):
      result = result + self.offset
      processed_results.append([result])
      if result == expect:
        self.good += 1
    self.total += len(results)
    return processed_results

  def start(self):
    self.good = 0
    self.total = 0

  def finalize(self, results):
    results["good"] = self.good
    results["total"] = self.total


class OFANetAutoSinian(nn.Module):
  def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
    super(OFANetAutoSinian, self).__init__()
    self.first_conv = first_conv
    self.blocks = nn.ModuleList(blocks)
    self.feature_mix_layer = feature_mix_layer
    self.classifier = classifier
    self.reduce_mean = nn.AdaptiveAvgPool2d((1, 1))
  def forward(self, x):
    x = self.first_conv(x)
    for block in self.blocks:
        x = block(x)
    x = self.feature_mix_layer(x)
    x = self.reduce_mean(x)
    x = torch.squeeze(torch.squeeze(x, 3), 2)
    x = self.classifier(x)
    return x
  
def load_model(args):
  sys.path.append(args.model_builder_dir)
  from model_zoo import set_layer_from_config, MobileInvertedResidualBlock
  net_config = json.load(open('pretrained/%s/net.config' % args.model))
  first_conv = set_layer_from_config(net_config['first_conv'])
  feature_mix_layer = set_layer_from_config(net_config['feature_mix_layer'])
  classifier = set_layer_from_config(net_config['classifier'])
  blocks = []
  for block_config in net_config['blocks']:
      blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))
  net = OFANetAutoSinian(first_conv, blocks, feature_mix_layer, classifier)
  for m in net.modules():
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
      m.momentum = net_config['bn']['momentum']
      m.eps = net_config['bn']['eps']
  image_size = net_config['image_size']
  loaded = torch.load('pretrained/%s/init' % args.model, map_location='cpu')
  if 'state_dict' in loaded:
    init = loaded['state_dict']
  else:
    init = loaded
  new_init = OrderedDict()
  for k in init.keys():
    if 'module.' in k:
      new_init[k.split('module.')[1]] = init[k]
  if new_init:
    init = new_init
  net.load_state_dict(init)
  return net, image_size


class Item:
  """An item that we queue for processing by the thread pool."""

  def __init__(self, index, query_id, img, label=None, results=None):
    self.query_id = query_id
    self.img = img
    self.label = label
    self.results = results
    self.index = index


def create_thread(pool, target, args):
  worker = threading.Thread(target=target, args=args)
  worker.daemon = True
  pool.append(worker)
  worker.start()


class QueueRunner:
  def __init__(self, model, ds, threads, post_proc=None, batch_size=128):
    self.ds = ds
    self.model = model
    self.post_proc = post_proc
    self.take_accuracy = False
    self.batch_size = batch_size
    self.tasks = Queue(maxsize=threads * 4)
    self.cpuTasks = Queue(maxsize=threads * 8)
    self.workers = []
    self.cpuWorkers = []
    self.result_dict = {}
    self.ppStream = torch.cuda.Stream()

    for _ in range(threads):
      create_thread(self.workers, self.handle_tasks,
                    (self.tasks, self.run_one_item))

    for _ in range(threads * 2):
      create_thread(self.cpuWorkers, self.handle_tasks,
                    (self.cpuTasks, self.postprocess))

  def handle_tasks(self, tasks_queue, fn):
    """Worker thread."""
    while True:
      qitem = tasks_queue.get()
      if qitem is None:
        # None in the queue indicates the parent want us to exit
        tasks_queue.task_done()
        break
      fn(qitem)
      tasks_queue.task_done()

  def start_run(self, result_dict, take_accuracy):
    self.result_dict = result_dict
    self.take_accuracy = take_accuracy
    self.post_proc.start()

  def run_one_item(self, qitem):
    qitem.readyEvent = torch.cuda.Event()
    qitem.results = self.model.predict(qitem.img)
    qitem.readyEvent.record()
    self.cpuTasks.put(qitem)

  def postprocess(self, qitem):
    qitem.readyEvent.synchronize()
    with torch.cuda.stream(self.ppStream):
      qitem.results = qitem.results.cpu()
    qitem.results = qitem.results.numpy()
    processed_results = self.post_proc(qitem.results, qitem.label,
                                       self.result_dict)
    response_array_refs = []
    response = []
    for result, query_id in zip(processed_results, qitem.query_id):
      response_array = array.array("B", np.array(result, np.float32).tobytes())
      response_array_refs.append(response_array)
      bi = response_array.buffer_info()
      response.append(lg.QuerySampleResponse(query_id, bi[0], bi[1]))
    lg.QuerySamplesComplete(response)
    if qitem.index == self.batches - 1:
      self.finishTime = time.time()

  def enqueue(self, ids, indices):
    data, label = self.ds.get_samples()
    queryLen = len(ids)
    bs = self.batch_size
    assert (queryLen % self.batch_size == 0
            ), "queryLen=%d must be dividable by batch size!" % queryLen
    self.batches = queryLen // bs
    audit_test_4b = True if indices[0] == indices[1] else False
    for i in range(0, queryLen, bs):
      query_id = []
      for j in range(bs):
        query_id.append(ids[i + j])
      if audit_test_4b:
        self.ds.unload_query_samples(None)
        self.ds.load_query_samples([indices[0]])
        data, label = self.ds.get_samples()
        label = label * bs
        data = data.cuda()
        data = data.repeat(bs, 1, 1, 1)
      self.tasks.put(Item(i // bs, query_id, data, label))

  def finish(self):
    for _ in self.workers:
      self.tasks.put(None)
    for worker in self.workers:
      worker.join()
    for _ in self.cpuWorkers:
      self.cpuTasks.put(None)
    for worker in self.cpuWorkers:
      worker.join()


def add_results(final_results, name, result_dict, result_list, took, args):
  percentiles = [50., 80., 90., 95., 99., 99.9]
  if result_list:
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(
        ["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])
  else:
    buckets_str = 'N/A'

  result = {
      "took": took,
      "mean": np.mean(result_list) if result_list else 0,
      "percentiles": {str(k): v
                      for k, v in zip(percentiles, buckets)}
      if result_list else '',
      "qps": result_dict["total"] / took,
      "count": result_dict["total"],
      "good_items": result_dict["good"],
      "total_items": result_dict["total"],
  }
  acc_str = ""
  result["accuracy"] = 100. * result_dict["good"] / result_dict["total"]
  acc_str = ", acc={:.3f}%".format(result["accuracy"])

  final_results[name] = result

  log.info(
      "{} qps={:.2f}, bs={}, thread={}, mean={:.4f}, time={:.3f}{}, queries={}, tiles={}".
      format(name, result["qps"], args.batch_size, args.threads,
             result["mean"], took, acc_str, result_dict["total"], buckets_str))


def main():
  global last_timeing
  args = get_args()

  log.info(args)

  backend = BackendTensorRT()

  ds = Imagenet(
      data_path=args.dataset_path,
      use_cache=args.cache,
      batch_size=args.batch_size,
      image_size=args.image_size,
      calib_file='cal_image_list_option_%d.txt' % args.calib_file)

  model = backend.load(args, ds=ds)

  final_results = {
      "runtime": model.name(),
      "version": model.version(),
      "time": int(time.time()),
      "cmdline": str(args),
  }

  config = os.path.abspath(args.config)
  assert(os.path.exists(config)), "%s not existed!" % config
  user_config = os.path.abspath(args.user_config)
  assert(os.path.exists(user_config)), "%s not existed!" % user_config

  base_path = os.path.dirname(os.path.realpath(__file__))
  if args.output:
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

  post_proc = PostProcessCommon(offset=0)
  runner = QueueRunner(
      model, ds, args.threads, post_proc=post_proc, batch_size=args.batch_size)

  def issue_queries(ids, indices):
    runner.enqueue(ids, indices)

  def flush_queries():
    pass

  def process_latencies(latencies_ns):
    global last_timeing
    last_timeing = [t / NANO_SEC for t in latencies_ns]

  settings = lg.TestSettings()
  model_name = 'OFAnet-AutoSinian'
  settings.FromConfig(config, model_name, args.scenario)
  settings.FromConfig(user_config, model_name, args.scenario)

  if args.audit_test:
    audit_config_path = base_path + '/audit%s.config' % args.audit_test
    settings.FromConfig(audit_config_path, model_name, args.scenario)

  scenario = SCENARIO_MAP[args.scenario]
  settings.scenario = scenario
  settings.mode = lg.TestMode.PerformanceOnly
  if args.accuracy:
    settings.mode = lg.TestMode.AccuracyOnly

  sut = lg.ConstructFastSUT(issue_queries, flush_queries, process_latencies)
  qsl = lg.ConstructQSL(ds.get_item_count(), args.batch_size,
                        ds.load_query_samples, ds.unload_query_samples)

  log.info("starting {}".format(scenario))
  result_dict = {"good": 0, "total": 0, "scenario": str(scenario)}
  runner.start_run(result_dict, args.accuracy)
  start = time.time()
  lg.StartTest(sut, qsl, settings)
  post_proc.finalize(result_dict)
  add_results(final_results, "{}".format(scenario), result_dict, last_timeing,
              runner.finishTime - ds.last_loaded, args)

  runner.finish()
  lg.DestroyQSL(qsl)
  lg.DestroyFastSUT(sut)

  if args.output:
    with open("results.json", "w") as f:
      json.dump(final_results, f, sort_keys=True, indent=2)


if __name__ == "__main__":
  main()
