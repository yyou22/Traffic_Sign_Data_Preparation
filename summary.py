import csv

total = 12629

file = open('summary.csv','w')
writer = csv.writer(file)

writer.writerow(['nat', 'adv'])

for i in range(0, 3):

	with open('data_' + str(i) + '_nat.csv') as file_obj:

		reader_obj = csv.reader(file_obj)

		correct_nat = 0

		for row in reader_obj:
			if row[2] == row[3]:
				correct_nat += 1

	with open('data_' + str(i) + '_adv.csv') as file_obj:

		reader_obj = csv.reader(file_obj)

		correct_adv = 0

		for row in reader_obj:
			if row[2] == row[3]:
				correct_adv += 1

	writer.writerow([str(correct_nat), str(correct_adv)])

