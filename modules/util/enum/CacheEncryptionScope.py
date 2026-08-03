from enum import Enum


class CacheEncryptionScope(Enum):
    ENCRYPTED_SOURCES = "ENCRYPTED_SOURCES"
    ALL = "ALL"

    def __str__(self):
        return self.value
