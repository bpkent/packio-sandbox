"""How I would serialize and deserialize without Packio."""

from datetime import datetime, timezone
import orjson
import pandas as pd
from pathlib import Path
from pydantic.dataclasses import dataclass
from uuid import uuid4, UUID


@dataclass(config=dict(arbitrary_types_allowed=True))
class CoolModel:
    """Brian's coolest ML model."""

    id: UUID
    documentation: str
    config: dict
    rmse: float
    trained_at: datetime
    df: pd.DataFrame

    def save(self, dir: Path):
        """Persist class object to disk."""

        if not dir.exists():
            dir.mkdir()

        metadata = {
            k: v for k, v in self.__dict__.items() if not isinstance(v, pd.DataFrame)
        }

        with open(dir / "metadata.json", "wb") as f:
            f.write(
                orjson.dumps(
                    metadata, option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2
                )
            )

        self.df.to_parquet(dir / "df.parquet")

    @classmethod
    def load(cls, dir):
        """Load persisted object from disk."""

        with open(dir / "metadata.json", "rb") as f:
            meta = orjson.loads(f.read())

        df = pd.read_parquet(dir / "df.parquet")

        model = cls(**meta, df=df)
        return model


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
