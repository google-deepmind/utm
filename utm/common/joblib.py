# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared joblib."""

from concurrent import futures
import functools
import queue
import time
import tqdm


def load_rules(fname):
  with open(fname, "r") as f:
    rule_list = f.readlines()
  rule_list = [s.strip() for s in rule_list]
  rule_list = [s for s in rule_list if s]
  print("number of rules", len(rule_list))
  return rule_list


def concat_rules(rule_list: list[str]) -> str:
  return ",".join(rule_list)


def run_job(rule, fn_prompt, model, result_pattern):
  """Run job."""
  key, value = rule.split(":")
  key = key.strip()
  value = value.strip()
  q = fn_prompt(key)
  # assume the result content is text string.
  result = model(q).content
  result = result.replace("<ctrl100>", "").strip()
  if ":" in result:
    result = result.split(":")[1].strip()
  if result != value:
    if result_pattern is None:
      return (result, False)
    m = result_pattern.findall(result)
    if not m or m[0] != value:
      return (result, False)
  return (result, True)


def main_loop(selected_rules, fn_prompt, result_pattern,
              model, num_workers=4, sleep=0, max_fail=100000):
  fail_count = 0
  count = 0
  worker = functools.partial(
      run_job, fn_prompt=fn_prompt, model=model,
      result_pattern=result_pattern)
  failed_rules = []
  if num_workers == 0:
    for rule in tqdm.tqdm(selected_rules):
      count += 1
      result, valid = worker(rule)
      if not valid:
        print("[true rule]", rule, "[model output]", result)
        fail_count += 1
        failed_rules.append(rule)
        if fail_count > max_fail:
          break
      if sleep > 0:
        time.sleep(sleep)
  else:
    thread_pool = futures.ThreadPoolExecutor(max_workers=num_workers)
    jobs = queue.Queue()
    pos = min(num_workers, len(selected_rules))
    for i in range(pos):
      rule = selected_rules[i]
      jobs.put((rule, thread_pool.submit(worker, rule)))
    pbar = tqdm.tqdm(total=len(selected_rules))
    while not jobs.empty():
      rule, job_handle = jobs.get()
      try:
        response = job_handle.result(timeout=30)
      except TimeoutError:
        job_handle.cancel()
        jobs.put((rule, thread_pool.submit(worker, rule)))
        continue
      result, valid = response
      count += 1
      pbar.update(1)
      if not valid:
        print("[true rule]", rule, "[model output]", result)
        fail_count += 1
        failed_rules.append(rule)
        if fail_count > max_fail:
          break
      if pos < len(selected_rules):
        rule = selected_rules[pos]
        jobs.put((rule, thread_pool.submit(worker, rule)))
        pos += 1

  print("total", count, "fail", fail_count)
  return failed_rules
