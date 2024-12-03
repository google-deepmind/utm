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

import abc
import base64
from collections.abc import Sequence
import dataclasses
import enum
from functools import cached_property  # pylint: disable=g-importing-member
import os
from typing import Any, Optional, Union

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import HarmBlockThreshold
from vertexai.generative_models import HarmCategory
from vertexai.generative_models import Image


@dataclasses.dataclass
class UserContent:
  """User content."""

  class CType(enum.IntEnum):
    STRING = 0
    IMAGE = 1

  ctype: CType
  value: str
  format: Optional[str] = None

  def __len__(self):
    return len(self.value)

  def __str__(self) -> str:
    if self.ctype == UserContent.CType.IMAGE:
      return f'(Image formatted as {self.format})\n'
    else:
      return self.value


@dataclasses.dataclass
class PromptPair:
  """Prompt pair."""

  user: Union[str, UserContent, Sequence[UserContent]]
  system: Optional[str] = None

  def __len__(self):
    tot = 0
    if self.system:
      tot += len(self.system)
    if isinstance(self.user, str) or isinstance(self.user, UserContent):
      tot += len(self.user)
    else:
      assert isinstance(self.user, Sequence)
      tot += sum([len(x) for x in self.user])
    return tot

  def __str__(self) -> str:
    if isinstance(self.user, Sequence):
      user_str = ''.join([str(x) for x in self.user])
    else:
      user_str = str(self.user)
    return f'System: {self.system}\nUser: {user_str}'


class LLM(abc.ABC):
  """LLM interface."""

  def __init__(self, model_name):
    self.model_name = model_name

  @abc.abstractmethod
  def __call__(self, prompt: PromptPair):
    """Issue the LLM call."""

  def _format_image(self, c: UserContent):
    raise NotImplementedError

  def _format_content(self, content):
    """Formatting."""
    if isinstance(content, str):
      content = [content]
    if not isinstance(content, Sequence):
      content = [content]
    result = []
    for c in content:
      if isinstance(c, str):
        result.append({'type': 'text', 'text': c})
      else:
        assert isinstance(c, UserContent)
        if c.ctype == UserContent.CType.STRING:
          result.append({'type': 'text', 'text': c.value})
        else:
          assert c.ctype == UserContent.CType.IMAGE
          result.append(self._format_image(c))
    return result


@dataclasses.dataclass
class LLMResponse(abc.ABC):

  prompt: PromptPair

  @cached_property
  @abc.abstractmethod
  def content(self) -> str:
    """Get the generated content in str format."""

  @cached_property
  @abc.abstractmethod
  def cost(self):
    """Get the cost of generating this response."""


@dataclasses.dataclass
class VertexGeminiResponse(LLMResponse):

  response: Any

  @cached_property
  def content(self):
    return self.response.text

  @cached_property
  def cost(self):
    return 0.0


class VertexGemini(LLM):
  """Gemini api."""

  def __init__(self, model_name):
    if model_name == 'pro-001':
      model_name = 'gemini-1.5-pro-001'
    elif model_name == 'flash-001':
      model_name = 'gemini-1.5-flash-001'
    elif model_name == 'pro-002':
      model_name = 'gemini-1.5-pro-002'
    elif model_name == 'flash-002':
      model_name = 'gemini-1.5-flash-002'
    super().__init__(model_name=model_name)
    vertexai.init(
        project=os.environ.get('GCP_PROJECT_ID'),
        location=os.environ.get('GCP_LOCATION'))
    # pylint: disable=line-too-long
    self.safety_settings = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    }
    # pylint: enable=line-too-long

  def __call__(self, prompt: PromptPair):
    model = GenerativeModel(
        self.model_name,
        generation_config=dict(
            max_output_tokens=64,
            candidate_count=1,
            temperature=0.0,
        ),
        system_instruction=prompt.system,
        safety_settings=self.safety_settings
    )
    content = self._format_content(prompt.user)
    return VertexGeminiResponse(
        prompt=prompt,
        response=model.generate_content(
            [elem[elem['type']] for elem in content], stream=False),
    )

  def _format_image(self, c: UserContent):
    image_bytes = base64.b64decode(c.value)
    return {'type': 'image', 'image': Image.from_bytes(image_bytes)}


if __name__ == '__main__':
  gemini_model = VertexGemini('pro-001')
  test_query = PromptPair(
      user='Hello!', system=None
  )
  print(gemini_model(test_query))
