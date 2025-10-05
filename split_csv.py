import csv
import os
import sys


def split_csv_by_columns(source_path: str, output_dir: str = "result") -> None:
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Input file not found: {source_path}")

    os.makedirs(output_dir, exist_ok=True)

    with open(source_path, newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError("Input CSV is empty") from None

        if len(headers) < 2:
            raise ValueError("CSV needs at least one x column and one y column")

        x_header = headers[0]
        y_headers = headers[1:]

        writers = []
        files = []
        try:
            for y_header in y_headers:
                safe_name = f"{y_header}.csv"
                output_path = os.path.join(output_dir, safe_name)
                outfile = open(output_path, "w", newline="", encoding="utf-8")
                writer = csv.writer(outfile)
                writer.writerow(['JD', 'gp_mean'])
                writers.append(writer)
                files.append(outfile)

            for row in reader:
                if len(row) != len(headers):
                    raise ValueError("Row length does not match header length")
                x_value = row[0]
                for idx, writer in enumerate(writers, start=1):
                    writer.writerow([x_value, row[idx]])
        finally:
            for f in files:
                f.close()


if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        cmd = os.path.basename(sys.argv[0])
        print(f"Usage: python {cmd} <input_csv> [output_dir]", file=sys.stderr)
        sys.exit(1)

    input_csv = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) == 3 else "result"

    split_csv_by_columns(input_csv, output_directory)
