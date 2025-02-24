from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from typing import List, Optional, Union


class InteractionCIPlotter:
    def __init__(
        self,
        regression_model: sm.regression.linear_model.RegressionResultsWrapper,
        dataframe: pd.DataFrame,
        output_path: str,
        interaction_term: str,
        explanatory_variable: str,
        dependent_variable: str,
        control_variables: Optional[List[str]] = None,
    ):
        """
        Initialize the InteractionCIPlotter class.

        Parameters:
        - regression_model: The regression model used for predictions.
        - dataframe: The dataframe containing the data.
        - output_path: Path to save the plot.
        - interaction_term: The interaction term in the model.
        - explanatory_variable: The explanatory variable in the model.
        - dependent_variable: The dependent variable in the model.
        - control_variables: List of control variables in the model.
        """
        self.regression_model = regression_model
        self.dataframe = dataframe
        self.output_path = output_path
        self.interaction_term = interaction_term
        self.explanatory_variable = explanatory_variable
        self.dependent_variable = dependent_variable
        self.control_variables = control_variables if control_variables else []

    def generate_predictions(
        self, explanatory_range: np.ndarray, interaction_value: Union[int, float, str]
    ) -> pd.DataFrame:
        """
        Generate predictions for the given explanatory range and interaction value.

        Parameters:
        - explanatory_range: Range of values for the explanatory variable.
        - interaction_value: Value of the interaction term.

        Returns:
        - DataFrame with predictions and confidence intervals.
        """
        control_means = {
            control_var: self.dataframe[control_var].mean()
            for control_var in self.control_variables
        }

        prediction_data = {
            self.explanatory_variable: explanatory_range,
            self.interaction_term: interaction_value,
            **control_means,
        }

        prediction_df = pd.DataFrame(prediction_data)

        predictions = self.regression_model.get_prediction(
            prediction_df
        ).summary_frame()
        prediction_df["predicted_mean"] = predictions["mean"]
        prediction_df["lower_bound"] = predictions["mean_ci_lower"]
        prediction_df["upper_bound"] = predictions["mean_ci_upper"]

        return prediction_df

    def plot_predictions(self, predictions_all: pd.DataFrame) -> None:
        """
        Plot the predictions with confidence intervals.

        Parameters:
        - predictions_all: DataFrame containing all predictions and confidence intervals.
        """
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=predictions_all,
            x=self.explanatory_variable,
            y="predicted_mean",
            hue=self.interaction_term,
            palette="tab10",
            linewidth=2,
        )

        unique_interactions = predictions_all[self.interaction_term].unique()
        for interaction_value in unique_interactions:
            interaction_data = predictions_all[
                predictions_all[self.interaction_term] == interaction_value
            ]
            plt.fill_between(
                interaction_data[self.explanatory_variable],
                interaction_data["lower_bound"],
                interaction_data["upper_bound"],
                alpha=0.2,
                label=None,
            )

        plt.xlabel(self.explanatory_variable.replace("_", " ").title())
        plt.ylabel(f"predicted_{self.dependent_variable}".replace("_", " ").title())
        plt.title(
            f"Predicted {self.dependent_variable.replace('_', ' ').title()} vs {self.explanatory_variable.replace('_', ' ').title()} by {self.interaction_term.replace('_', ' ').title()}"
        )
        if self.control_variables:
            control_vars_str = ", ".join(self.control_variables)
            plt.suptitle(
                f"Predicted at the mean of control variables: {control_vars_str}",
                fontsize=10,
                y=0.90,
            )
            plt.subplots_adjust(top=0.85)
        plt.legend(title=self.interaction_term.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.show()
        print(f"Plot saved to {Path(self.output_path).resolve()}")

    def create_plot(self, smoothness: int = 1000) -> None:
        """
        Create and save the plot with predictions and confidence intervals.

        Parameters:
        - smoothness: Number of points in the explanatory range for smooth plotting.
        """
        explanatory_range = np.linspace(
            self.dataframe[self.explanatory_variable].min(),
            self.dataframe[self.explanatory_variable].max(),
            smoothness,
        )
        unique_interactions = self.dataframe[self.interaction_term].unique()

        predictions_all = pd.DataFrame()
        for interaction_value in unique_interactions:
            prediction_df = self.generate_predictions(
                explanatory_range, interaction_value
            )
            predictions_all = pd.concat(
                [predictions_all, prediction_df], ignore_index=True
            )

        self.plot_predictions(predictions_all)


class ScatterLowessCIPlotter:
    def __init__(
        self,
        title: str,
        xlabel: str,
        ylabel: str,
        x_var: str,
        y_var: str,
        data: pd.DataFrame,
    ):
        """
        Initialize the ScatterLowessCIPlotter class.

        Parameters:
        - title: Title of the plot.
        - xlabel: Label for the x-axis.
        - ylabel: Label for the y-axis.
        - x_var: Name of the x variable.
        - y_var: Name of the y variable.
        - data: DataFrame containing the data.
        """
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x_var = x_var
        self.y_var = y_var
        self.data = data.copy()

    def plot(self, lowess_frac: float = 0.2, save_path: Optional[str] = None) -> None:
        """
        Plot scatter plot with Lowess line and confidence intervals.

        Parameters:
        - lowess_frac: Fraction of data used for Lowess smoothing.
        - save_path: Path to save the plot, if provided.
        """
        data = self.data
        # Calculate Lowess smoother
        lowess = sm.nonparametric.lowess
        lowess_results = lowess(data[self.y_var], data[self.x_var], frac=lowess_frac)

        # Extract Lowess results
        lowess_x = lowess_results[:, 0]
        lowess_y = lowess_results[:, 1]

        # Estimate residuals
        residuals = data[self.y_var] - np.interp(data[self.x_var], lowess_x, lowess_y)
        std_error = np.std(residuals)

        # Calculate confidence intervals
        ci = 1.96 * std_error  # 95% confidence interval
        lowess_ci_lower = lowess_y - ci
        lowess_ci_upper = lowess_y + ci

        # Plot scatter plot with Lowess line and confidence intervals
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=self.x_var, y=self.y_var, data=data, alpha=0.3)
        plt.plot(lowess_x, lowess_y, color="red", label="Lowess")
        plt.fill_between(
            lowess_x,
            lowess_ci_lower,
            lowess_ci_upper,
            color="red",
            alpha=0.2,
            label="Lowess CI (std. norm, 95%)",
        )
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()


class ViolinPlotter:
    def __init__(
        self,
        data: pd.DataFrame,
        days_of_week_column: str,
        confounding_variables: List[str],
    ):
        """
        Initialize the ViolinPlotter class.

        Parameters:
        - data: The dataset to plot.
        - days_of_week_column: The column with days of the week categories.
        - confounding_variables: A list of variable names to explore.
        """
        self.data = data
        self.days_of_week_column = days_of_week_column
        self.confounding_variables = confounding_variables
        # Ensure days of the week are ordered starting from Monday
        self.data[self.days_of_week_column] = pd.Categorical(
            self.data[self.days_of_week_column],
            categories=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
            ordered=True,
        )

    def plot(self, figsize: tuple = (15, 10), save_path: Optional[str] = None) -> None:
        """
        Generate and display a single plot with subplots for each variable using Seaborn and Matplotlib.

        Parameters:
        - figsize: Size of the figure.
        - save_path: If provided, saves the figure to this path.
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        for i, variable in enumerate(self.confounding_variables):
            sns.violinplot(
                ax=axes[i],
                data=self.data,
                x=self.days_of_week_column,
                y=variable,
                inner="quartile",
            )
            # axes[i].set_title(
            #     f"Violin Plot of {variable} by Day of the Week", fontsize=12
            # )
            axes[i].set_xlabel("Day of the Week")
            axes[i].set_ylabel(variable.replace("_", " ").title())

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(
            "Violin Plots of Confounding Variables by Day of the Week", fontsize=16
        )
        if save_path:
            plt.savefig(save_path)
        plt.show()
