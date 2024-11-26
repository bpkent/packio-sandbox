"""How I would serialize and deserialize without Packio."""

from datetime import datetime, timezone
import packio
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from uuid import uuid4, UUID


@packio(dir="test_model")  # Gives me `save` and `load` methods automagically, like a dataclass.
@dataclass
class CoolModel:
    """Brian's coolest ML model."""

    id: UUID
    documentation: str
    config: dict
    rmse: float
    trained_at: datetime
    df: pd.DataFrame


m = CoolModel(
    id=uuid4(),
    documentation="Brian's cool ML model.",
    config=dict(lr=0.01, num_trees=100),
    rmse=0.13,
    trained_at=datetime.now(tz=timezone.utc),
    df=pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 7]}),
)

m.save(Path("test_model"))
m2 = CoolModel.load(Path("test_model"))

assert id(m) != id(m2)
assert m.id == m2.id
assert m.trained_at == m2.trained_at
assert m.documentation == m2.documentation
assert m.config == m2.config
assert m.rmse == m2.rmse
pd.testing.assert_frame_equal(m.df, m2.df)
