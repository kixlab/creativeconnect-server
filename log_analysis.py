# Read jsonl file
import json

userId = [
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "P8",
    "P9",
    "P10",
    "P11",
    "P13",
    "P14",
    "P15",
    "P16",
]
conditions = ["TREATMENT", "CONTROL"]

f_w = open("log_analysis.csv", "w")
f_w.write("id,condition,num,time,rate\n")

for id in userId:
    for condition in conditions:
        print(id + "-" + condition)
        file_path = "logs/" + id + "-" + condition + ".jsonl"
        log_data = []
        with open(file_path, "r") as f:
            for line in f:
                log_data.append(json.loads(line))

        sketch_data = []
        start_time = log_data[0]["timestamp"]
        num = 1
        for i in range(len(log_data)):
            if log_data[i]["action"] == "finishSketch":
                f_w.write(
                    id
                    + ","
                    + condition
                    + ","
                    + str(num)
                    + ","
                    + str(log_data[i]["timestamp"] - start_time)
                    + ","
                    + str(log_data[i]["data"]["rate"])
                    + "\n"
                )
                sketch_data.append(
                    {
                        "time": log_data[i]["timestamp"] - start_time,
                        "rate": log_data[i]["data"]["rate"],
                    }
                )
                num += 1
                start_time = log_data[i]["timestamp"]

        print(sketch_data)
