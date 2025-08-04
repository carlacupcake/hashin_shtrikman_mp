"""optimization_result_visualizer.py."""
import numpy as np
import plotly.graph_objects as go

from ..genetic_algorithm import GeneticAlgorithmResult, Member
from ..utilities import get_headers


class OptimizationResultVisualizer():
    """
    Visualization class for plotting results of genetic algorithm optimization.
    """

    def __init__(self, ga_result: GeneticAlgorithmResult) -> None:
        self.ga_params = ga_result.algo_parameters
        self.opt_params = ga_result.optimization_params
        self.ga_result = ga_result


    def get_table_of_best_designs(self, rows: int = 10):
        """
        Retrieves a table of the top-performing designs from the final population.

        Args:
            rows (int, optional)
            - The number of top designs to retrieve.

        Returns:
            table_of_best_designs (ndarray)
        """

        [unique_values, unique_eff_props, unique_costs] = self.ga_result.final_population.get_unique_designs()
        table_of_best_designs = np.hstack((unique_values[0:rows, :],
                                           unique_eff_props[0:rows, :],
                                           unique_costs[0:rows].reshape(-1, 1)))
        return table_of_best_designs


    def print_table_of_best_designs(self, rows: int = 10):
        """
        Generates and displays a formatted table of the top-performing designs.

        Args:
            rows (int, optional)

        Returns:
            plotly.graph_objects.Figure
        """

        table_data = self.get_table_of_best_designs(rows)
        headers = get_headers(self.opt_params.num_materials,
                              self.opt_params.property_categories)

        header_color   = "lavender"
        odd_row_color  = "white"
        even_row_color = "lightgrey"
        if rows % 2 == 0:
            multiplier  = int(rows/2)
            cells_color = [[odd_row_color, even_row_color]*multiplier]
        else:
            multiplier  = int(np.floor(rows/2))
            cells_color = [[odd_row_color, even_row_color]*multiplier]
            cells_color.append(odd_row_color)

        fig = go.Figure(data=[go.Table(
            columnwidth = 1000,
            header = dict(
                values=headers,
                fill_color=header_color,
                align="left",
                font=dict(size=12),
                height=30
            ),
            cells = dict(
                values=[table_data[:, i] for i in range(table_data.shape[1])],
                fill_color=cells_color,
                align="left",
                font=dict(size=12),
                height=30,
            )
        )])

        # Update layout for horizontal scrolling
        fig.update_layout(
            title="Optimal Properties Recommended by Genetic Algorithm",
            title_font_size=20,
            title_x=0.2,
            margin=dict(l=0, r=0, t=40, b=0),
            height=400,
            autosize=True
        )

        return fig


    def plot_optimization_results(self):
        """
        Generates a plot visualizing the optimization convergence over generations.

        Returns:
            plotly.graph_objects.Figure
        """

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(self.ga_params.num_generations)),
            y=self.ga_result.avg_parent_costs,
            mode="lines",
            name="Avg. of top 10 performers"
        ))

        fig.add_trace(go.Scatter(
            x=list(range(self.ga_params.num_generations)),
            y=self.ga_result.lowest_costs,
            mode="lines",
            name="Best costs"
        ))

        fig.update_layout(
            title="Convergence of Genetic Algorithm",
            title_x=0.25,
            xaxis_title="Generation",
            yaxis_title="Cost",
            legend=dict(
                font=dict(size=14),
                x=1,
                y=1,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(255, 255, 255, 0.5)"
            ),
            title_font_size=24,
            xaxis_title_font_size=20,
            yaxis_title_font_size=20,
            width=600,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        return fig


    def plot_cost_func_contribs(self):
        """
        Generates a pie chart visualizing the contributions of different factors
        to the cost function for the best design found by the genetic algorithm.

        Returns:
            plotly.graph_objects.Figure
        """

        # Get the best design
        best_design = Member(optimization_params=self.opt_params,
                             values=self.ga_result.final_population.values[0],
                             ga_params=self.ga_params)
        cost, costs_eff_props, costs_cfs = best_design.get_cost(include_cost_breakdown=True)
        print(f"Cost: {cost}, Number effective properties: {len(costs_eff_props)}, Number of concentration factors: {len(costs_cfs)}") # temporary

        # Scale the costs of the effective properties and concentration factors
        # Scale according to weights from GAParams
        scaled_costs_eff_props = 1/2 * self.ga_params.weight_eff_prop * costs_eff_props
        scaled_costs_cfs = 1/2 * self.ga_params.weight_conc_factor * costs_cfs

        # Labels for the pie chart
        eff_prop_labels = []
        cf_labels = []
        for category in self.opt_params.property_categories:
            for property in self.opt_params.property_docs[category]:
                eff_prop_labels.append(f"eff. {property}")
                for m in range(self.opt_params.num_materials):
                    if property == "bulk_modulus":
                        cf_labels.append(f"cf hydrostatic stress, mat {m}")
                    elif property == "shear_modulus":
                        cf_labels.append(f"cf deviatoric stress, mat {m}")
                    else:
                        cf_labels.append(f"cf load on {property}, mat {m}")
                        cf_labels.append(f"cf response from {property}, mat {m}")
        labels = eff_prop_labels + cf_labels

        # Combine the data and labels for the eff props and concentration factors
        scaled_costs_eff_props = np.array(scaled_costs_eff_props)
        scaled_costs_cfs = np.array(scaled_costs_cfs)
        cost_func_contribs = np.concatenate((scaled_costs_eff_props, scaled_costs_cfs))

        # Create the pie chart figure
        fig = go.Figure(data=[go.Pie(labels=labels, values=cost_func_contribs,
                                     textinfo="percent",
                                     insidetextorientation="radial",
                                     hole=.25)])

        fig.update_layout(
            title_text="Cost Function Contributions",
            showlegend=True,
        )

        # Display the chart
        return fig
