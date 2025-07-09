import sqlite3

conn = sqlite3.connect("./estate.db")
cursor = conn.cursor()


def parse_line(line: str) -> list:
    res = []

    try:
        for (i, val) in enumerate(line.split(",")):
            if i == 1:
                res.append(int(val))
            elif i == 2:
                splitted_duration_string = val.split(" ")
                duration = int(
                    splitted_duration_string[-1]) - int(splitted_duration_string[0])

                res.append(duration)
            else:
                res.append(float(val))
        return res
    except Exception as e:
        raise e


def create_dataset():

    try:
        dataset = open("../linear_model/dataset/taiwan_real_estate.csv")
        first = True

        for line in dataset.readlines():
            if first:
                first = False
                continue

            try:
                values = parse_line(line)
                print("Values: ", values)

                cursor.execute(
                    "INSERT INTO estate (distance_to_market, num_convenience_stores, house_age, price_per_msq) VALUES (?, ?, ?, ?)", values)
                conn.commit()
            except Exception as parse_line_error:
                print(parse_line_error)
                continue
    except FileNotFoundError:
        print("Dataset not found")
        return


def main():
    cursor.execute("SELECT * FROM estate")
    values = cursor.fetchall()
    print("values: ", values)


if __name__ == "__main__":
    main()
