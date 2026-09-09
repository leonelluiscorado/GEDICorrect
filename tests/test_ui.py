import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


UI_PATH = Path(__file__).parents[1] / "src" / "gedicorrect" / "ui.py"


class DashboardTests(unittest.TestCase):
    def run_dashboard(self):
        dashboard = AppTest.from_file(str(UI_PATH), default_timeout=20)
        dashboard.run()
        self.assertFalse(dashboard.exception)
        return dashboard

    @staticmethod
    def widget_by_label(widgets, label):
        return next(widget for widget in widgets if widget.label == label)

    def test_random_candidate_controls_are_reactive(self):
        dashboard = self.run_dashboard()
        number_inputs = {widget.label: widget for widget in dashboard.number_input}

        self.assertNotIn("Random points", number_inputs)
        self.assertFalse(number_inputs["Grid size"].disabled)
        self.assertFalse(number_inputs["Grid step (m)"].disabled)
        self.assertFalse(number_inputs["Time window"].disabled)

        random_mode = self.widget_by_label(dashboard.checkbox, "Use random candidate points")
        random_mode.set_value(True)
        dashboard.run()
        number_inputs = {widget.label: widget for widget in dashboard.number_input}

        self.assertIn("Random points", number_inputs)
        self.assertIn("Maximum radius (m)", number_inputs)
        self.assertIn("Minimum distance (m)", number_inputs)
        self.assertFalse(number_inputs["Random points"].disabled)
        self.assertNotIn("Grid size", number_inputs)
        self.assertNotIn("Grid step (m)", number_inputs)
        self.assertNotIn("Time window", number_inputs)

        correction_mode = self.widget_by_label(dashboard.selectbox, "Correction mode")
        correction_mode.set_value("beam")
        dashboard.run()
        number_inputs = {widget.label: widget for widget in dashboard.number_input}
        random_mode = self.widget_by_label(dashboard.checkbox, "Use random candidate points")

        self.assertTrue(random_mode.disabled)
        self.assertFalse(random_mode.value)
        self.assertNotIn("Random points", number_inputs)
        self.assertFalse(number_inputs["Grid size"].disabled)
        self.assertFalse(number_inputs["Grid step (m)"].disabled)
        self.assertFalse(number_inputs["Time window"].disabled)


if __name__ == "__main__":
    unittest.main()
