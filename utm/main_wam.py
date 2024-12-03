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

"""Calling cloud api."""

from collections.abc import Sequence
import random

from absl import app
from absl import flags
from utm.common import joblib
from utm.common.llm import PromptPair
from utm.common.llm import VertexGemini


_VERSION = flags.DEFINE_string("version", "21", "version of the rules")
_MODEL = flags.DEFINE_string("model", "pro-001", "gemini model verion")
_NUM_WORKERS = flags.DEFINE_integer("num_workers", 4, "num workers")
_SLEEP = flags.DEFINE_integer("sleep", 0, "sleep time between requests")
_RULE_COPIES = flags.DEFINE_integer(
    "rule_copies", 1, "repeat rules in the prompt")


FLAGS = flags.FLAGS


def reformat_rule(rule) -> str:
  return rule.strip()


result_pattern = None


def prepare_prompt(selected_rules_, extra_rules=None):
  """Prepares prompt."""
  selected_rules = selected_rules_[:]
  if _RULE_COPIES.value > 1:
    rule_rep = selected_rules[:]
    for _ in range(_RULE_COPIES.value - 1):
      selected_rules += rule_rep
  if extra_rules:
    selected_rules = selected_rules + extra_rules
  prompt_tpl = "".join([
      "Given a dictionary of comma separated key->value pairs followed by a ",
      "key, respond with the corresponding value and nothing else.\n",
      "Given the dictionary {{",
      "{rules}",
      "}}, what is the value for "
      ])
  print(prompt_tpl.format(
      rules=joblib.concat_rules(selected_rules[:4]) + ".." +
      joblib.concat_rules(extra_rules[-4:])))
  print("totally", len(selected_rules), "in prompt")
  return prompt_tpl.format(rules=joblib.concat_rules(selected_rules))


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  model = VertexGemini(_MODEL.value)
  rule_file = f"rules{_VERSION.value}.txt"
  rule_list = joblib.load_rules(rule_file)

  extra_rules = []
  if _MODEL.value == "pro-001":
    if _VERSION.value == "21":
      extra_rules = ["fGgJ:nJ", "gJkF:gJ", "vJqB:aB"]
    elif _VERSION.value == "23":
      extra_rules = ["r3r6:d7", "x0g0:b7", "k0f7:f7", "x0g0:b7", "k0f7:f7"]
  elif _MODEL.value == "pro-002":
    if _VERSION.value == "21":
      extra_rules = ["vJvI:vJ", "gJyG:gJ", "vJsJ:vJ", "vJgI:vJ", "aHvJ:dM",
                     "vJdG:vJ", "vJhI:vJ", "kGgJ:vJ", "vJwI:vJ", "vJvI:vJ",
                     "vJsJ:vJ", "vJgI:vJ", "vJhI:vJ", "vJjH:oD", "vJsJ:vJ",
                     "vJhI:vJ", "vJwD:zG", "fGgJ:nJ", "rKvJ:vJ", "jIvJ:jI",
                     "vJhI:vJ", "vJsF:vJ", "vJhI:vJ"]
    elif _VERSION.value == "23":
      extra_rules = ["v3f7:f7", "y7f7:f7", "y4n6:r6", "l8u8:h1", "g0q2:h1",
                     "e8j0:h1", "v3m5:f7", "k0f7:f7", "a9m5:h1", "l8u8:h1",
                     "o2m2:f7", "k0f7:f7"]
  random.seed(1)
  random.shuffle(rule_list)
  selected_rules = [reformat_rule(rule) for rule in rule_list]

  while True:
    prompt = prepare_prompt(selected_rules, extra_rules)
    fn_prompt = lambda key: PromptPair(user=prompt + key.strip(), system=None)
    failed_rules = joblib.main_loop(
        selected_rules, fn_prompt, result_pattern, model,
        num_workers=_NUM_WORKERS.value, sleep=_SLEEP.value)
    if not failed_rules:
      break
    extra_rules += failed_rules


if __name__ == "__main__":
  app.run(main)
