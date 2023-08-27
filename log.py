from flask import Blueprint, jsonify, request, Response
import json
import traceback


def log(userId, action, data, timestamp):
    with open("./logs/" + userId + ".jsonl", "a") as f:
        f.write(
            json.dumps({"action": action, "data": data, "timestamp": timestamp}) + "\n"
        )


def create_log_api() -> Blueprint:
    api = Blueprint("log", __name__)

    @api.route("/add", methods=["POST"])
    def add_log():
        try:
            userId = request.json["userId"]
            action = request.json["action"]
            data = request.json["data"]
            timestamp = request.json["timestamp"]

            log(userId, action, data, timestamp)

            return Response(status=200)
        except Exception as e:
            traceback.print_exc()
            return Response(status=500)

    @api.route("/get", methods=["GET"])
    def get_log():
        # get user id in query
        userId = request.args.get("userId")
        if userId is None:
            return Response(status=400)
        else:
            with open("./log/" + userId + ".jsonl", "r") as f:
                lines = f.readlines()
                return jsonify(list(map(lambda x: json.loads(x), lines)))

    return api
