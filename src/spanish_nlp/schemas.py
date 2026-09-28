from pydantic import BaseModel, Field


class TopicRequest(BaseModel):
    documents: list[str] = Field(min_length=1)
    n_topics: int = Field(default=5, ge=1)


class TopicMemberResponse(BaseModel):
    text: str
    distance: float


class TopicResponse(BaseModel):
    topic_id: int
    representative: str
    members: list[TopicMemberResponse]


class SentimentRequest(BaseModel):
    documents: list[str] = Field(min_length=1)
    include_sentences: bool = True


class PolarityResponse(BaseModel):
    positive: float
    negative: float
    net: float


class SentencePolarityResponse(BaseModel):
    text: str
    score: PolarityResponse


class DocumentPolarityResponse(BaseModel):
    score: PolarityResponse
    sentences: list[SentencePolarityResponse]
