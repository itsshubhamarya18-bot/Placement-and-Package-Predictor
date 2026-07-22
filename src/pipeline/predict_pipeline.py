import os
import pickle
import pandas as pd
import numpy as np


class CustomData:

    def __init__(
        self,
        age,
        gender,
        cgpa,
        branch,
        college_tier,
        internships_count,
        projects_count,
        certifications_count,
        coding_skill_score,
        aptitude_score,
        communication_skill_score,
        logical_reasoning_score,
        hackathons_participated,
        github_repos,
        linkedin_connections,
        mock_interview_score,
        attendance_percentage,
        backlogs,
        extracurricular_score,
        leadership_score,
        volunteer_experience,
        sleep_hours,
        study_hours_per_day
    ):

        self.age = age
        self.gender = gender
        self.cgpa = cgpa
        self.branch = branch
        self.college_tier = college_tier
        self.internships_count = internships_count
        self.projects_count = projects_count
        self.certifications_count = certifications_count
        self.coding_skill_score = coding_skill_score
        self.aptitude_score = aptitude_score
        self.communication_skill_score = communication_skill_score
        self.logical_reasoning_score = logical_reasoning_score
        self.hackathons_participated = hackathons_participated
        self.github_repos = github_repos
        self.linkedin_connections = linkedin_connections
        self.mock_interview_score = mock_interview_score
        self.attendance_percentage = attendance_percentage
        self.backlogs = backlogs
        self.extracurricular_score = extracurricular_score
        self.leadership_score = leadership_score
        self.volunteer_experience = volunteer_experience
        self.sleep_hours = sleep_hours
        self.study_hours_per_day = study_hours_per_day

    def get_data_as_dataframe(self):

        custom_data_input_dict = {

            "age": [self.age],
            "gender": [self.gender],
            "cgpa": [self.cgpa],
            "branch": [self.branch],
            "college_tier": [self.college_tier],
            "internships_count": [self.internships_count],
            "projects_count": [self.projects_count],
            "certifications_count": [self.certifications_count],
            "coding_skill_score": [self.coding_skill_score],
            "aptitude_score": [self.aptitude_score],
            "communication_skill_score": [self.communication_skill_score],
            "logical_reasoning_score": [self.logical_reasoning_score],
            "hackathons_participated": [self.hackathons_participated],
            "github_repos": [self.github_repos],
            "linkedin_connections": [self.linkedin_connections],
            "mock_interview_score": [self.mock_interview_score],
            "attendance_percentage": [self.attendance_percentage],
            "backlogs": [self.backlogs],
            "extracurricular_score": [self.extracurricular_score],
            "leadership_score": [self.leadership_score],
            "volunteer_experience": [self.volunteer_experience],
            "sleep_hours": [self.sleep_hours],
            "study_hours_per_day": [self.study_hours_per_day]

        }

        return pd.DataFrame(custom_data_input_dict)


class PredictPipeline:

    def __init__(self):

        model_path = os.path.join("models", "placement_model.pkl")
        package_model_path = os.path.join("models", "package_model.pkl")
        scaler_path = os.path.join("models", "scaler.pkl")
        feature_path = os.path.join("models", "feature_columns.pkl")

        self.placement_model = pickle.load(open(model_path, "rb"))
        self.package_model = pickle.load(open(package_model_path, "rb"))
        self.scaler = pickle.load(open(scaler_path, "rb"))
        self.feature_columns = pickle.load(open(feature_path, "rb"))

    def predict(self, processed_data):

        # -------------------------------
            # One-Hot Encoding
            # -------------------------------
            processed_data = pd.get_dummies(
                processed_data,
                drop_first=True
            )

            # -------------------------------
            # Match Training Features
            # -------------------------------
            processed_data = processed_data.reindex(
                columns=self.feature_columns,
                fill_value=0
            )

            # -------------------------------
            # Feature Scaling
            # -------------------------------
            processed_data = self.scaler.transform(processed_data)

            # -------------------------------
            # Placement Prediction
            # -------------------------------
            placement_prediction = self.placement_model.predict(
                processed_data
            )[0]

            placement_probability = self.placement_model.predict_proba(
                processed_data
            )[0][1]

            probability = round(
                placement_probability * 100,
                2
            )

            # -------------------------------
            # Confidence Level
            # -------------------------------
            if probability >= 80:

                confidence = "High"

            elif probability >= 60:

                confidence = "Medium"

            else:

                confidence = "Low"

            # -------------------------------
            # Recommendation List
            # -------------------------------
            recommendations = []

            if probability < 40:

                recommendations.extend([

                    "Improve your coding skills.",
                    "Complete more internships.",
                    "Increase your project portfolio.",
                    "Practice aptitude questions daily.",
                    "Participate in hackathons."

                ])

            elif probability < 70:

                recommendations.extend([

                    "Practice mock interviews.",
                    "Earn additional certifications.",
                    "Improve communication skills.",
                    "Strengthen GitHub profile.",
                    "Enhance logical reasoning."

                ])

            else:

                recommendations.extend([

                    "Keep improving your DSA skills.",
                    "Continue participating in contests.",
                    "Maintain your CGPA.",
                    "Apply to product-based companies.",
                    "Prepare for HR interviews."

                ])

            # -------------------------------
            # If Student is NOT Placed
            # -------------------------------
            if placement_prediction == 0:

                return {

                    "placement": "Not Placed",

                    "probability": probability,

                    "confidence": confidence,

                    "package": None,

                    "message": "Your current profile needs improvement before placement.",

                    "recommendation": recommendations

                }

            # -------------------------------
            # Student Placed
            # -------------------------------

            # -------------------------------
            # Package Prediction
            # -------------------------------

            predicted_package = self.package_model.predict(
                processed_data
            )[0]

            # Prevent negative package prediction
            predicted_package = max(0, predicted_package)

            predicted_package = round(predicted_package, 2)

            # -------------------------------
            # Package Category
            # -------------------------------

            if predicted_package >= 20:

                package_level = "Excellent"

            elif predicted_package >= 12:

                package_level = "Very Good"

            elif predicted_package >= 8:

                package_level = "Good"

            elif predicted_package >= 5:

                package_level = "Average"

            else:

                package_level = "Entry Level"

            # -------------------------------
            # Success Message
            # -------------------------------

            message = (
                "Congratulations! Based on your profile, "
                "you have a good chance of getting placed."
            )

            # -------------------------------
            # Additional Recommendations
            # -------------------------------

            recommendations.append(
                "Continue improving your technical and interview skills."
            )

            recommendations.append(
                "Build strong real-world projects for your resume."
            )

            # -------------------------------
            # Final Response
            # -------------------------------

            result = {

                "placement": "Placed",

                "probability": probability,

                "confidence": confidence,

                "package": predicted_package,

                "package_level": package_level,

                "message": message,

                "recommendation": recommendations

            }

            return result