from pydantic import BaseModel, Field, ConfigDict


class RecurrentRegionConfig(BaseModel):
    name: str
    num_units: int = Field(gt=0)
    base_firing: float = 0.0
    init: float = 0.0
    sign: str = "pos"
    parent_region: str | None = None
    learnable_bias: bool = False
    device: str = "cuda"


class RecurrentConnectionConfig(BaseModel):
    src_region: str
    dst_region: str
    sparsity: float | None = Field(default=None, ge=0, le=1)


class InputRegionConfig(BaseModel):
    name: str
    num_units: int = Field(gt=0)
    sign: str = "pos"
    device: str = "cuda"


class InputConnectionConfig(BaseModel):
    src_region: str
    dst_region: str
    sparsity: float | None = Field(default=None, ge=0, le=1)


class mRNNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recurrent_regions: list[RecurrentRegionConfig] = Field(default_factory=list)
    recurrent_connections: list[RecurrentConnectionConfig] = Field(default_factory=list)
    input_regions: list[InputRegionConfig] = Field(default_factory=list)
    input_connections: list[InputConnectionConfig] = Field(default_factory=list)
