install:
	pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

build:
	docker build -t sod .

run:
	docker run -p $(PORT):8501 sod

local_run:
	streamlit run sod/app.py