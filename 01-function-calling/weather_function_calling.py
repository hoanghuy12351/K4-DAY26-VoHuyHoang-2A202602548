"""Minh hoạ FUNCTION CALLING thuần với Google Gemini SDK.

Các tool `get_weather` và `get_forecast` được định nghĩa schema thủ công VÀ thực thi ngay trong
chính file app này. Model chỉ QUYẾT ĐỊNH gọi tool nào; app mới là nơi chạy.

Cách chạy:
    pip install -r ../requirements.txt
    $env:GEMINI_API_KEY = "..."
    python weather_function_calling.py
"""

import json
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from google import genai
from google.genai import types

client = genai.Client()

MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý thời tiết thân thiện, trả lời bằng tiếng Việt tự nhiên. "
    "Nếu tool trả lỗi, hãy nói rõ chưa lấy được dữ liệu thời tiết. "
    "Dùng emoji phù hợp (🌧️ 🌤️ 💨 💧). "
    "Tóm tắt ngắn gọn, dễ hiểu. Sau phần dự báo, thêm mục 'Lời khuyên:' "
    "với 1-2 hành động cụ thể dựa trên nhiệt độ và xác suất mưa của các ngày được hỏi. "
    "Chỉ dùng dữ liệu tool cung cấp; không suy đoán kiểu mưa, trời nắng hay cảm giác dễ chịu "
    "khi tool không có các thông tin đó. Xác suất mưa không có nghĩa là chắc chắn sẽ mưa."
)

# 1. App tự định nghĩa schema của tool
get_weather_declaration = types.FunctionDeclaration(
    name="get_weather",
    description="Lấy thời tiết hiện tại của một thành phố",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(
                type=types.Type.STRING, description="Tên thành phố"
            )
        },
        required=["city"],
    ),
)

get_forecast_declaration = types.FunctionDeclaration(
    name="get_forecast",
    description="Lấy dự báo thời tiết 1 đến 3 ngày cho một thành phố từ Open-Meteo",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "city": types.Schema(type=types.Type.STRING, description="Tên thành phố"),
            "days": types.Schema(
                type=types.Type.INTEGER,
                description="Số ngày dự báo, từ 1 đến 3; mặc định là 2",
            ),
        },
        required=["city"],
    ),
)

TOOLS = [
    types.Tool(function_declarations=[get_weather_declaration, get_forecast_declaration])
]


# 2. Tool thời tiết hiện tại giữ dữ liệu mẫu để minh họa function calling.
def _city_key(name: str) -> str:
    text = unicodedata.normalize("NFD", name.casefold().replace("đ", "d"))
    return "".join(char for char in text if not unicodedata.combining(char))


def get_weather(city: str) -> str:
    """Trả về thời tiết (mock) của *city*. Dùng làm tool cho model."""
    mock_data = {
        "Hà Nội": {
            "nhiệt_độ": "29°C",
            "thời_tiết": "trời mưa nhẹ",
            "độ_ẩm": "82%",
            "gió": {"hướng": "Đông Nam", "tốc_độ": "12 km/h"},
        },
        "Hồ Chí Minh": {
            "nhiệt_độ": "33°C",
            "thời_tiết": "mưa rào",
            "độ_ẩm": "75%",
            "gió": {"hướng": "Tây Nam", "tốc_độ": "15 km/h"},
        },
        "Đà Nẵng": {
            "nhiệt_độ": "30°C",
            "thời_tiết": "nhiều mây",
            "độ_ẩm": "78%",
            "gió": {"hướng": "Đông", "tốc_độ": "10 km/h"},
        },
    }
    default = {"nhiệt_độ": "28°C", "thời_tiết": "không có dữ liệu chi tiết"}
    return json.dumps({"city": city, **mock_data.get(city, default)}, ensure_ascii=False)


def get_forecast(city: str, days: int = 2) -> str:
    """Lấy dự báo 1-3 ngày từ Open-Meteo; không cần key thời tiết."""
    if not isinstance(city, str) or not city.strip():
        return json.dumps({"error": "Cần tên thành phố."}, ensure_ascii=False)
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 3:
        return json.dumps({"error": "days phải là số nguyên từ 1 đến 3."}, ensure_ascii=False)
    city = city.strip()

    try:
        if _city_key(city) in {"ha noi", "hanoi"}:
            # Tọa độ Hà Nội tránh lỗi ghép tên sang Hà Tĩnh.
            place = {"name": "Hà Nội", "country": "Vietnam", "latitude": 21.0283, "longitude": 105.8542}
        else:
            query = urlencode({"name": city, "count": 10})
            with urlopen(f"https://geocoding-api.open-meteo.com/v1/search?{query}", timeout=10) as response:
                places = json.load(response).get("results", [])
            name = _city_key(city.split(",", 1)[0].strip())
            matches = [
                item for item in places
                if _city_key(item["name"]).removesuffix(" city") == name
            ]
            if len(matches) != 1:
                return json.dumps({"error": "Địa điểm chưa rõ; hãy thêm quốc gia."}, ensure_ascii=False)
            place = matches[0]

        query = urlencode({
            "latitude": place["latitude"], "longitude": place["longitude"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": days, "timezone": "auto",
        })
        with urlopen(f"https://api.open-meteo.com/v1/forecast?{query}", timeout=10) as response:
            daily = json.load(response)["daily"]
        forecast = [
            {"date": date, "min_c": daily["temperature_2m_min"][i],
             "max_c": daily["temperature_2m_max"][i],
             "rain_probability_percent": daily["precipitation_probability_max"][i]}
            for i, date in enumerate(daily["time"])
        ]
        if len(forecast) != days:
            return json.dumps({"error": "Số ngày dự báo trả về không khớp."}, ensure_ascii=False)
        return json.dumps({"city": place["name"], "country": place["country"], "forecast": forecast}, ensure_ascii=False)
    except HTTPError as exc:
        return json.dumps({"error": f"Open-Meteo trả HTTP {exc.code}."}, ensure_ascii=False)
    except (URLError, TimeoutError, KeyError, TypeError, ValueError, IndexError):
        return json.dumps({"error": "Không lấy được dữ liệu dự báo."}, ensure_ascii=False)


def run(prompt: str) -> str:
    """Gửi *prompt* tới Gemini, tự động xử lý function calling và trả về câu trả lời cuối."""
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]
    tool_functions = {"get_weather": get_weather, "get_forecast": get_forecast}

    # 3. Gọi model — model quyết định có gọi tool hay không
    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=TOOLS,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    # 4. Vòng lặp: nếu model yêu cầu tool, app TỰ THỰC THI rồi đưa kết quả trả lại
    while resp.function_calls:
        # Thêm phản hồi của model vào lịch sử hội thoại
        contents.append(resp.candidates[0].content)

        function_responses = []
        for fc in resp.function_calls:
            print(f"  [model yêu cầu] {fc.name}({fc.args})")
            tool = tool_functions.get(fc.name)
            result = (
                tool(**fc.args)
                if tool
                else json.dumps({"error": f"Tool không hỗ trợ: {fc.name}"}, ensure_ascii=False)
            )
            print(f"  [app thực thi]  -> {result}")
            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": result}
                )
            )

        # Gửi kết quả tool trả về cho model
        contents.append(types.Content(role="user", parts=function_responses))
        resp = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=TOOLS,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )

    # 5. Model tổng hợp câu trả lời cuối
    return resp.text


if __name__ == "__main__":
    question = input("Bạn muốn hỏi gì về thời tiết? ").strip()
    if not question:
        raise SystemExit("Bạn cần nhập câu hỏi.")
    print(f"User: {question}\n")
    print("Trả lời:", run(question))
