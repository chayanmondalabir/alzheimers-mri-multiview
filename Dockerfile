FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# download the ANTsPyNet skull-stripping weights once, at build time
RUN python -c "import numpy as np, ants, antspynet; im=ants.from_numpy(np.random.rand(96,96,96).astype('float32')); antspynet.brain_extraction(im, modality='t1')" || true
COPY app ./app
COPY scripts ./scripts
# trained weights come from the GitHub Release (see scripts/fetch_weights.py)
RUN python scripts/fetch_weights.py
ENV PYTHONUNBUFFERED=1
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.maxUploadSize=200", "--browser.gatherUsageStats=false"]
