from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = CustomData(

            age=int(request.form.get("age")),

            gender=request.form.get("gender"),

            cgpa=float(request.form.get("cgpa")),

            branch=request.form.get("branch"),

            college_tier=(request.form.get("college_tier")),

            internships_count=int(request.form.get("internships_count")),

            projects_count=int(request.form.get("projects_count")),

            certifications_count=int(request.form.get("certifications_count")),

            coding_skill_score=int(request.form.get("coding_skill_score")),

            aptitude_score=int(request.form.get("aptitude_score")),

            communication_skill_score=int(request.form.get("communication_skill_score")),

            logical_reasoning_score=int(request.form.get("logical_reasoning_score")),

            hackathons_participated=int(request.form.get("hackathons_participated")),

            github_repos=int(request.form.get("github_repos")),

            linkedin_connections=int(request.form.get("linkedin_connections")),

            mock_interview_score=int(request.form.get("mock_interview_score")),

            attendance_percentage=float(request.form.get("attendance_percentage")),

            backlogs=int(request.form.get("backlogs")),

            extracurricular_score=int(request.form.get("extracurricular_score")),

            leadership_score=int(request.form.get("leadership_score")),

            volunteer_experience=(request.form.get("volunteer_experience")),

            sleep_hours=float(request.form.get("sleep_hours")),

            study_hours_per_day=float(request.form.get("study_hours_per_day"))

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