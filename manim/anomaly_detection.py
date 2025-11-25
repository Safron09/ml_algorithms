from manim import *

# -------------------------------------------------------------
# SCENE 1 — INTRO + AXES + BASIC EXPLANATION
# -------------------------------------------------------------
class Scene1_Intro(Scene):
    def construct(self):

        # Title in center
        title = Text("Anomaly Detection", font_size=60)
        self.play(FadeIn(title))
        self.wait(0.7)

        # Move to top
        self.play(title.animate.to_edge(UP))
        self.wait(0.5)

        # Draw X-Y axes (0..10)
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=6,
            y_length=6,
            tips=False
        ).shift(DOWN * 0.5)

        labels = axes.get_axis_labels(
            Text("X"), Text("Y")
        )

        self.play(Create(axes), FadeIn(labels))
        self.wait(0.5)

        # Bottom explanation
        explanation = Text(
            "Anomaly detection identifies unusual data points\n"
            "that deviate from normal patterns in ML.",
            font_size=26
        ).to_edge(DOWN)

        self.play(FadeIn(explanation))
        self.wait(2)


# -------------------------------------------------------------
# SCENE 2 — CENTROIDS + DATA POINTS + FORMULA + MOVEMENT
# -------------------------------------------------------------
class Scene2_Centroids(Scene):
    def construct(self):

        # Bring axes back
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=6,
            y_length=6,
            tips=False
        ).shift(DOWN * 0.5)

        labels = axes.get_axis_labels(Text("X"), Text("Y"))
        self.play(Create(axes), FadeIn(labels))
        self.wait(0.5)

        # Three centroids
        centroid1 = Dot(axes.c2p(2, 8), color=YELLOW)
        centroid2 = Dot(axes.c2p(7, 7), color=YELLOW)
        centroid3 = Dot(axes.c2p(4, 2), color=YELLOW)

        centroids = VGroup(centroid1, centroid2, centroid3)
        self.play(FadeIn(centroids))
        self.wait(0.5)

        # Data clusters
        cluster1 = VGroup(
            Dot(axes.c2p(1.5, 7.5), color=BLUE),
            Dot(axes.c2p(2.5, 8.2), color=BLUE),
            Dot(axes.c2p(1.8, 8.4), color=BLUE),
            Dot(axes.c2p(2.2, 7.8), color=BLUE)
        )

        cluster2 = VGroup(
            Dot(axes.c2p(6.5, 7.2), color=BLUE),
            Dot(axes.c2p(7.5, 7.4), color=BLUE),
            Dot(axes.c2p(7, 7.8), color=BLUE),
            Dot(axes.c2p(6.8, 6.6), color=BLUE)
        )

        cluster3 = VGroup(
            Dot(axes.c2p(3.5, 1.8), color=BLUE),
            Dot(axes.c2p(4.5, 1.6), color=BLUE),
            Dot(axes.c2p(4.2, 2.4), color=BLUE),
            Dot(axes.c2p(3.8, 2.2), color=BLUE)
        )

        all_points = VGroup(cluster1, cluster2, cluster3)
        self.play(FadeIn(all_points))
        self.wait(0.8)

        # Explanation text
        explanation = Text(
            "Centroids are the centers of normal data clusters.\n"
            "They are updated by computing the mean of nearby points.",
            font_size=24
        ).to_edge(DOWN)

        formula = Text(
            "μ = (1/m) · Σ xᵢ",
            font_size=32
        ).next_to(explanation, UP)

        self.play(FadeIn(explanation), FadeIn(formula))
        self.wait(1)

        # Animate centroid movement (simulating convergence)
        self.play(
            centroid1.animate.move_to(axes.c2p(2.0, 8.0)),
            centroid2.animate.move_to(axes.c2p(7.1, 7.1)),
            centroid3.animate.move_to(axes.c2p(4.0, 2.0)),
            run_time=2
        )
        self.wait(2)


# -------------------------------------------------------------
# SCENE 3 — FINAL VIEW + WHY ANOMALY DETECTION MATTERS
# -------------------------------------------------------------
class Scene3_Why(Scene):
    def construct(self):

        # Title (small)
        header = Text("Why Anomaly Detection?", font_size=40)
        self.play(FadeIn(header.scale(1.1)))
        self.wait(0.5)

        # Explanation
        explanation = Text(
            "Used to find fraud, failures, unusual patterns,\n"
            "and rare events that deviate from normal behavior.",
            font_size=28
        ).next_to(header, DOWN, buff=0.6)

        self.play(FadeIn(explanation))
        self.wait(2)
