.PHONY: run dev docker-build docker-run test

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t claude-sequence-probability-scorer:latest .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env claude-sequence-probability-scorer:latest

test:
	pytest -q
