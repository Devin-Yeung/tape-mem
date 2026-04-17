from dataclasses import dataclass


@dataclass
class ProviderConfig:
    model: str
    base_url: str
    api_key: str
