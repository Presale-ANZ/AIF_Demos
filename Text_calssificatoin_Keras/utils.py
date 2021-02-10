import csv


class Util:
    @staticmethod
    def read_csv(file):

        bodies = []
        labels = []

        with open(file, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter=",")
            next(reader)
            for row in reader:
                labels.append(row[0].replace(" ", "_"))
                bodies.append(row[1])
        return bodies, labels
