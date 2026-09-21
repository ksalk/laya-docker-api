import uvicorn

from .main import parse_args

args = parse_args()
uvicorn.run(
    "app.main:app",
    host=args.host,
    port=args.port,
    workers=1,
    log_level="info",
)
