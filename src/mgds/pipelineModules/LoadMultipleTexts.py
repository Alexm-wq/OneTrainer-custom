import os

from mgds.PipelineModule import PipelineModule
from mgds.crypto import read_source_text
from mgds.pipelineModuleTypes.RandomAccessPipelineModule import RandomAccessPipelineModule


class LoadMultipleTexts(PipelineModule, RandomAccessPipelineModule):
    def __init__(self, path_in_name: str, texts_out_name: str):
        super().__init__()
        self.path_in_name = path_in_name
        self.texts_out_name = texts_out_name

    def length(self) -> int:
        return self._get_previous_length(self.path_in_name)

    def get_inputs(self) -> list[str]:
        return [self.path_in_name]

    def get_outputs(self) -> list[str]:
        return [self.texts_out_name]

    def get_item(self, variation: int, index: int, requested_name: str = None) -> dict:
        path = self._get_previous_item(variation, self.path_in_name, index)
        texts = []
        if os.path.exists(path):
            try:
                texts = [line.strip() for line in read_source_text(path).splitlines()]
            except FileNotFoundError:
                texts = [""]
            except UnicodeDecodeError as exc:
                raise RuntimeError(
                    f"Failed to load caption file '{path}': decrypted/plain data is not valid UTF-8."
                ) from exc
            except Exception:
                print("could not load text, it might be corrupted: " + path)
                raise
        texts = [text for text in texts if text != ""]
        if not texts:
            texts = [""]
        return {self.texts_out_name: texts}
