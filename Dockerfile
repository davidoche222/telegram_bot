FROM python:3.10-slim

# Install system dependencies required for TA-Lib
RUN apt-get update && apt-get install -y \
    build-essential \
    libtool \
    automake \
    autoconf \
    wget \
    ca-certificates \
    gcc \
    make \
    && rm -rf /var/lib/apt/lists/*

# Download and install TA-Lib C library (robust method)
RUN wget -q https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make -j2 \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /app

# Upgrade pip tools first
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
