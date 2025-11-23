from manim import *
import numpy as np

config.background_color = "#0E0E2E"


class LogisticRegressionVisualization(Scene):

    def show_explanation(self, lines, color=WHITE, delay=3):
        text = VGroup(*[
            Text(line, font_size=32, color=color)
            for line in lines
        ]).arrange(DOWN).to_edge(LEFT).shift(DOWN * 1)

        self.play(FadeIn(text, shift=UP))
        self.wait(delay)
        self.play(FadeOut(text, shift=DOWN))

    def construct(self):

        # Title
        title = Text("Logistic Regression", font_size=54).to_edge(UP)
        title.set_color_by_gradient(BLUE, PURPLE)
        self.play(Write(title), run_time=2)

        # --- Explanation 1: what it is ---
        self.show_explanation([
            "Logistic Regression is used for classification.",
            "It predicts probabilities between 0 and 1.",
        ])

        # Axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            axis_config={"color": BLUE},
            tips=False
        ).shift(DOWN * 0.3)

        x_label = Text("x1", font_size=28).next_to(axes.x_axis, DOWN)
        y_label = Text("x2", font_size=28).next_to(axes.y_axis, LEFT)
        labels = VGroup(x_label, y_label)

        self.play(Create(axes), FadeIn(labels), run_time=2)

        # Positive & negative example points
        positive_points = [
            axes.c2p(1.5, 1), axes.c2p(2, 0.5), axes.c2p(1, 1.5)
        ]
        negative_points = [
            axes.c2p(-2, -1), axes.c2p(-1.5, -0.5), axes.c2p(-1, -1.5)
        ]

        pos_dots = VGroup(*[Dot(p, color=GREEN) for p in positive_points])
        neg_dots = VGroup(*[Dot(p, color=RED) for p in negative_points])

        self.play(FadeIn(pos_dots), FadeIn(neg_dots), run_time=2)

        # Linear boundary line
        line = Line(
            start=axes.c2p(-2, 2),
            end=axes.c2p(2, -2),
            color=YELLOW,
            stroke_width=4
        )
        line_label = Text("Linear Boundary", font_size=32).next_to(line, RIGHT)

        self.play(Create(line), Write(line_label), run_time=2)

        self.wait(1)

        # --- Explanation 2: why useful ---
        self.show_explanation([
            "Why Logistic Regression?",
            "• Fast and simple",
            "• Works well for binary classification",
            "• Produces probabilities, not just labels",
        ])

        # Transition to sigmoid
        sigmoid_title = Text("Sigmoid Function", font_size=44).to_edge(UP)
        sigmoid_title.set_color_by_gradient(BLUE, PURPLE)
        self.play(Transform(title, sigmoid_title))

        # Sigmoid axes (no LaTeX)
        sig_axes = Axes(
            x_range=[-6, 6, 2],
            y_range=[0, 1, 0.2],
            x_length=7,
            y_length=3,
            axis_config={"color": BLUE},
            tips=False,
        ).shift(DOWN * 1.5)

        self.play(Create(sig_axes), run_time=2)

        # Sigmoid curve
        sigmoid_graph = sig_axes.plot(
            lambda x: 1 / (1 + np.exp(-x)),
            color=YELLOW
        )
        self.play(Create(sigmoid_graph), run_time=3)

        z_label = Text("z", font_size=28).next_to(sig_axes.x_axis, DOWN)
        sigma_label = Text("sigma(z)", font_size=28).next_to(sig_axes.y_axis, LEFT)
        self.play(FadeIn(z_label), FadeIn(sigma_label))

        # --- Explanation 3: pros & cons ---
        self.show_explanation([
            "Pros:",
            "Interpretable",
            "Fast",
            "Works for small datasets",
        ])

        self.show_explanation([
            "Cons:",
            "Only linear boundaries",
            "Bad for complex patterns",
            "Sensitive to outliers",
        ])

        # Probability prediction
        prob_text = Text("Prediction = sigmoid(w · x + b)", font_size=40)
        prob_text.set_color(YELLOW)
        prob_text.next_to(sig_axes, DOWN, buff=0.5)

        self.play(Write(prob_text), run_time=2)

        # --- Explanation 4: when to use it ---
        self.show_explanation([
            "Use Logistic Regression when:",
            "• Problem is binary classification",
            "• Relationship is roughly linear",
            "• You need probabilities",
        ])

        self.show_explanation([
            "Avoid Logistic Regression if:",
            "• Data is highly non-linear",
            "• Many feature interactions",
            "• Classes overlap heavily",
        ])

        self.wait(2)
