FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY bot.py /app/bot.py
COPY cache_market_data.py /app/cache_market_data.py
COPY trading_bot /app/trading_bot
COPY README.md /app/README.md

RUN mkdir -p /app/.runtime /app/.cache/market_data

CMD ["python3", "bot.py", "paper-watch", "--symbol", "ETHUSDT", "--preset", "hybrid_1h", "--quote-order-qty", "50", "--poll-seconds", "60", "--portfolio-name", "demo", "--initial-cash", "1000"]
