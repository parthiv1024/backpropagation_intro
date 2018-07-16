from numpy import *

# y = mx + b
# m is gradient, b is y-intercept
def compute_error_for_given_points(b, m, points):
	totalError = 0
	for i in range(0, len(points)):
		x = points[i][0]
		y = points[i][1]
		# Error for a point = (actual y - predicted y) ** 2
		totalError += (y - (m * x + b)) ** 2
	return totalError/float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(len(points)):
		x = points[i][0]
		y = points[i][1]
		# Each individual gradient acts as a vector for that point and the sum of these vectors shows us which direction to move in
		# (1/N) d/db (y - (mx + b)) ** 2 = 2/N * (y-(mx + b)) * (-1) [Chain Rule]
		b_gradient += (-2/N) * (y - ((m_current * x) + b_current))
		# (1/N) d/dm (y - (mx + b)) ** 2 = 2/N * (y-(mx + b)) * (x) [Chain Rule]
		m_gradient += (-2/N) * x * (y - ((m_current * x) + b_current))
	new_b = b_current - (learningRate * b_gradient)
	new_m = m_current - (learningRate * m_gradient)
	return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
	b = starting_b
	m = starting_m
	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learning_rate)
	return [b, m]

def run():
	points = genfromtxt("data.csv", delimiter=",")
	learning_rate = 0.0001
	initial_b = 0
	initial_m = 0
	num_iterations = 1000
	print("Starting gradient descent at b={0}, m={1} and error={2}".format(initial_b, initial_m, compute_error_for_given_points(initial_b, initial_m, points)))
	print("Running")
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print("After {0} iterations b={1}, m={2} and error={3}".format(num_iterations, b, m, compute_error_for_given_points(b, m, points)))

if __name__ == '__main__':
	run()