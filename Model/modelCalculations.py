import numpy as np


class LinearRegression:
    def __init__(self, w, b, x, y, iteration, alpha, m):
        self.w = w
        self.b = b
        self.x = x
        self.y = y
        self.iteration = iteration
        self.alpha = alpha
        self.m = m

    def calculate_prediction(self):
        y_pred = np.dot(self.x, self.w) + self.b
        return y_pred

    def calculate_cost_function(self, y_pred):
        error = y_pred - self.y
        cost = (1 / (2 * self.m)) * np.sum(error**2)
        return cost

    def calculate_gradiant(self, y_pred):
        error = y_pred - self.y
        dw = (1 / self.m) * np.dot(self.x.T, error)
        db = (1 / self.m) * np.sum(error)
        return dw, db

    def calculate_gradiant_descent(self, dw, db):
        self.w = self.w - self.alpha * dw
        self.b = self.b - self.alpha * db

    def linear_regression(self):
        cost_history = []
        iteration_history = []

        for i in range(self.iteration):
            y_pred = self.calculate_prediction()
            dw, db = self.calculate_gradiant(y_pred)

            if (
                np.any(np.isnan(dw))
                or np.any(np.isnan(db))
                or np.any(np.isinf(dw))
                or np.any(np.isinf(db))
            ):
                print(f"!!! Error: NaN/Inf detected at iteration {i}. Lower Alpha!")
                break

            self.calculate_gradiant_descent(dw, db)

            if i % 100 == 0:
                cost = self.calculate_cost_function(y_pred)
                cost_history.append(cost)
                iteration_history.append(i)
                print(f"Iteration {i}: Cost {cost:.6f}")

        y_pred_final = self.calculate_prediction()

        print(f"\nb,w found by gradient descent: {self.b:.2f}, {np.round(self.w, 2)}")
        for j in range(self.m):
            print(f"prediction: {y_pred_final[j]:.2f}, target value: {self.y[j]}")

        return self.w, self.b, cost_history, iteration_history

    def predict(self, x_input):
        """Make a prediction for new input data."""
        return np.dot(x_input, self.w) + self.b
