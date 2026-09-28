from fastapi import FastAPI, HTTPException

from spanish_nlp.config import Settings
from spanish_nlp.resources import load_sentiwordnet, load_stopwords
from spanish_nlp.runtime import load_nlp
from spanish_nlp.schemas import (
    DocumentPolarityResponse,
    PolarityResponse,
    SentencePolarityResponse,
    SentimentRequest,
    TopicMemberResponse,
    TopicRequest,
    TopicResponse,
)
from spanish_nlp.sentiment import analyze_document
from spanish_nlp.topic import detect_topics

settings = Settings()
app = FastAPI(
    title="Spanish NLP Toolkit",
    version="0.1.0",
    description="Topic clustering and lexical sentiment analysis for Spanish text.",
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "spacy_model": settings.spacy_model,
        "sentiment_lexicon_configured": settings.swn_path is not None,
    }


@app.post("/topics", response_model=list[TopicResponse])
def topics(request: TopicRequest) -> list[TopicResponse]:
    try:
        nlp = load_nlp(settings.spacy_model)
        stopwords = load_stopwords(settings.stopwords_path)
        clusters = detect_topics(
            nlp,
            request.documents,
            n_topics=request.n_topics,
            random_state=settings.random_state,
            extra_stopwords=stopwords,
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return [
        TopicResponse(
            topic_id=cluster.topic_id,
            representative=cluster.representative,
            members=[
                TopicMemberResponse(text=member.text, distance=member.distance)
                for member in cluster.members
            ],
        )
        for cluster in clusters
    ]


@app.post("/sentiment", response_model=list[DocumentPolarityResponse])
def sentiment(request: SentimentRequest) -> list[DocumentPolarityResponse]:
    if settings.swn_path is None:
        raise HTTPException(
            status_code=503,
            detail="Set SWN_PATH to a compatible Spanish SentiWordNet TSV resource.",
        )

    try:
        nlp = load_nlp(settings.spacy_model)
        lexicon = load_sentiwordnet(settings.swn_path)
        stopwords = load_stopwords(settings.stopwords_path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    results = []
    for text in request.documents:
        analysis = analyze_document(
            nlp,
            text,
            lexicon,
            include_sentences=request.include_sentences,
            extra_stopwords=stopwords,
        )
        results.append(
            DocumentPolarityResponse(
                score=PolarityResponse(
                    positive=analysis.score.positive,
                    negative=analysis.score.negative,
                    net=analysis.score.net,
                ),
                sentences=[
                    SentencePolarityResponse(
                        text=sentence.text,
                        score=PolarityResponse(
                            positive=sentence.score.positive,
                            negative=sentence.score.negative,
                            net=sentence.score.net,
                        ),
                    )
                    for sentence in analysis.sentences
                ],
            )
        )
    return results
