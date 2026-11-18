import json
import logging
import os
import re

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import APIError, Groq
from pydantic import BaseModel, Field
from requests import RequestException

from kakao_search import search_kakao_places

load_dotenv()

logger = logging.getLogger("uvicorn.error")
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key, timeout=20.0, max_retries=0) if groq_api_key else None
groq_model = "openai/gpt-oss-120b"
station_pattern = re.compile(r"(?<![가-힣])(?!(?:지역|구역|전역|광역|권역))([가-힣]+역)")

default_origins = "http://localhost:3000,http://heartbreakkid.co.kr,https://heartbreakkid.co.kr"
allowed_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", default_origins).split(",")
    if origin.strip()
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    district: str
    previous_places: list[str] = Field(default_factory=list)
    previous_query: str = ""


class PlaceInfo(BaseModel):
    name: str
    category: str
    address: str
    lat: float
    lng: float
    is_registered: bool
    description: str


class ChatResponse(BaseModel):
    reply: str
    show_map: bool = False
    places: list[PlaceInfo] = Field(default_factory=list)


class SearchIntent(BaseModel):
    keyword: str
    is_place_request: bool


@app.get("/healthz")
def healthcheck():
    return {"status": "ok"}


def groq_json(groq_client: Groq, prompt: str, message: str, temperature: float) -> str:
    completion = groq_client.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content or ""


def recommend_places(request: ChatRequest, search_message: str, keyword: str) -> ChatResponse:
    keyword = station_pattern.sub("", keyword).strip() or "맛집"
    station_match = station_pattern.search(search_message)

    if station_match:
        station_name = station_match.group(1)
        station_results = search_kakao_places(station_name, 15)
        station = next(
            (
                place for place in station_results
                if place["name"] == station_name
                or place["name"].startswith(station_name + " ")
            ),
            None,
        )

        if not station:
            return ChatResponse(reply=f"{station_name}의 위치를 찾지 못했어요.")

        actual_district = next(
            (
                district for district in ["노원구", "도봉구", "중랑구", "강북구", "성북구"]
                if district in station["address"]
            ),
            None,
        )
        if actual_district != request.district:
            return ChatResponse(
                reply=(
                    f"{station_name}은(는) {actual_district or '다른 지역'}에 위치해 있어요! 😊\n\n"
                    f"현재 선택하신 지역은 {request.district}입니다.\n\n"
                    "상단에서 지역을 변경해 주세요."
                )
            )

        results = search_kakao_places(
            keyword, 15, x=station["lng"], y=station["lat"], radius=1000
        )
    else:
        results = search_kakao_places(f"{request.district} {keyword}", 15)

    previous_names = set(request.previous_places)
    places = [
        PlaceInfo.model_validate(place)
        for place in results
        if request.district in place["address"] and place["name"] not in previous_names
    ][:3]

    if not places:
        return ChatResponse(reply="조건에 맞는 장소를 찾지 못했어요.\n\n지역이나 역 이름을 확인해 주세요.")

    recommendations = "\n\n".join(
        f"{index}. {place.name}\n위치: {place.address}"
        for index, place in enumerate(places, 1)
    )
    return ChatResponse(
        reply=f"추천 장소를 소개해 드릴게요! 😊\n\n{recommendations}",
        show_map=True,
        places=places,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if client is None:
        logger.error("GROQ_API_KEY is missing.")
        raise HTTPException(status_code=503, detail="채팅 서비스가 설정되지 않았습니다.")

    search_message = request.message
    if request.previous_query and any(
        word in request.message for word in ("다른", "더 추천", "또 추천", "새로운")
    ):
        search_message = request.previous_query

    try:
        intent = SearchIntent.model_validate_json(groq_json(
            client,
            """
            사용자의 질문을 분석하여 장소 검색어를 추출하세요.
            음식 종류나 카페, 학원 같은 장소 종류를 검색어로 사용하세요.
            특정 종류가 없으면 '맛집'을 사용하세요.
            음식 종류를 임의로 변경하지 마세요.
            장소 추천 요청이 아니면 is_place_request를 false로 설정하세요.
            JSON 형식: {"keyword": "검색어", "is_place_request": true}
            """,
            search_message,
            temperature=0,
        ))

        if not intent.is_place_request:
            content = groq_json(
                client,
                f"""
                당신은 친절한 지역 안내 AI 누비니입니다. 사용자가 선택한 지역은 {request.district}입니다.
                누비니는 전국을 누비고 다닌다는 뜻입니다.
                자연스러운 한국어 존댓말을 사용하고, 주제가 바뀔 때 빈 줄로 문단을 구분하세요.
                일반 대화에 답하고, 실제 확인된 장소 정보가 필요한 질문에는 장소를 지어내지 마세요.
                JSON 형식: {{"reply": "사용자에게 보여줄 답변"}}
                """,
                request.message,
                temperature=0.6,
            )
            return ChatResponse(reply=json.loads(content)["reply"])
    except (APIError, ValueError, TypeError, KeyError, IndexError) as exc:
        logger.warning("Groq chat failed (%s).", type(exc).__name__)
        return ChatResponse(reply="현재 AI 응답을 가져오지 못했어요. 잠시 후 다시 시도해 주세요.")

    try:
        return recommend_places(request, search_message, intent.keyword)
    except (RequestException, ValueError, TypeError, KeyError) as exc:
        logger.warning("Kakao place search failed (%s).", type(exc).__name__)
        return ChatResponse(reply="현재 카카오 장소 검색을 이용할 수 없어요. 잠시 후 다시 시도해 주세요.")
