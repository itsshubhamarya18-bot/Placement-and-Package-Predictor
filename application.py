from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    def safe_int(val, default=0):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return default
            
    def safe_float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    try:

        data = CustomData(

            age=safe_int(request.form.get("age")),

            gender=request.form.get("gender", "Male"),

            cgpa=safe_float(request.form.get("cgpa")),

            branch=request.form.get("branch", "CSE"),

            college_tier=request.form.get("college_tier", "Tier 1"),

            internships_count=safe_int(request.form.get("internships_count")),

            projects_count=safe_int(request.form.get("projects_count")),

            certifications_count=safe_int(request.form.get("certifications_count")),

            coding_skill_score=safe_int(request.form.get("coding_skill_score")),

            aptitude_score=safe_int(request.form.get("aptitude_score")),

            communication_skill_score=safe_int(request.form.get("communication_skill_score")),

            logical_reasoning_score=safe_int(request.form.get("logical_reasoning_score")),

            hackathons_participated=safe_int(request.form.get("hackathons_participated")),

            github_repos=safe_int(request.form.get("github_repos")),

            linkedin_connections=safe_int(request.form.get("linkedin_connections")),

            mock_interview_score=safe_int(request.form.get("mock_interview_score")),

            attendance_percentage=safe_float(request.form.get("attendance_percentage")),

            backlogs=safe_int(request.form.get("backlogs")),

            extracurricular_score=safe_int(request.form.get("extracurricular_score")),

            leadership_score=safe_int(request.form.get("leadership_score")),

            volunteer_experience=request.form.get("volunteer_experience", "No"),

            sleep_hours=safe_float(request.form.get("sleep_hours")),

            study_hours_per_day=safe_float(request.form.get("study_hours_per_day"))

        )

        processed_data = data.get_data_as_dataframe()

        prediction_pipeline = PredictPipeline()

        result = prediction_pipeline.predict(processed_data)

        return render_template(

            "index.html",

            result=result,

            form_data=request.form.to_dict()

        )

    except Exception as e:
        print("ERROR:", e)
        raise


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=5000, debug=True)