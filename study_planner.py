import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_study_plan(student_info, rag_manager=None, llm=None):
    name = student_info["name"]
    grade = student_info["grade"]
    field = student_info["field"]
    goal = student_info["goal"]
    daily_hours = student_info["daily_hours"]
    start_date = student_info["start_date"]
    subjects = student_info["subjects"]
    priorities = student_info["priorities"]
    notes = student_info["notes"]
    
    
    days = ["شنبه", "یکشنبه", "دوشنبه", "سه‌شنبه", "چهارشنبه", "پنج‌شنبه", "جمعه"]
    day_dates = []
    
    
    current_date = start_date
    for i in range(7):
        day_dates.append(current_date)
        current_date += timedelta(days=1)
    
    
    total_priority = sum(priorities)
    time_ratios = [p / total_priority for p in priorities]
    
    
    subject_daily_hours = [daily_hours * ratio for ratio in time_ratios]
    
    
    study_plan_data = []
    
    
    if rag_manager:
        query = f"""
        بهترین روش برنامه‌ریزی مطالعه برای یک دانش‌آموز {grade} رشته {field} که برای {goal} آماده می‌شود چیست؟
        لطفا توصیه‌هایی برای چیدمان دروس در طول هفته ارائه دهید.
        """
        
        
        recommendations = rag_manager.get_response(query)
        
        use_recommendations = True
    else:
        use_recommendations = False
        recommendations = ""
    
    
    for day_idx, day in enumerate(days):
        date = day_dates[day_idx]
        date_str = date.strftime("%Y-%m-%d")
        
        num_subjects_today = min(len(subjects), 4)
        
        if use_recommendations and "ریاضی" in recommendations.lower() and day_idx < 3:
            today_indices = []
            for idx, subject in enumerate(subjects):
                if "ریاضی" in subject:
                    today_indices.append(idx)
            
            
            remaining_slots = num_subjects_today - len(today_indices)
            if remaining_slots > 0:
                other_indices = [i for i in range(len(subjects)) if i not in today_indices]
                other_priorities = [priorities[i] for i in other_indices]
                selected_other = np.random.choice(
                    other_indices, 
                    size=min(remaining_slots, len(other_indices)), 
                    replace=False, 
                    p=np.array(other_priorities)/sum(other_priorities) if sum(other_priorities) > 0 else None
                )
                today_indices.extend(selected_other)
        else:
            today_indices = np.random.choice(
                range(len(subjects)), 
                size=min(num_subjects_today, len(subjects)), 
                replace=False, 
                p=np.array(priorities)/sum(priorities)
            )
        
        
        for idx in today_indices:
            subject = subjects[idx]
            hours = subject_daily_hours[idx] * (0.8 + 0.4 * np.random.random())  
            
            start_time = f"{15 + np.random.randint(0, 5)}:00"
            
            hours_int = int(hours)
            minutes_int = int((hours - hours_int) * 60)
            duration = f"{hours_int}:{minutes_int:02d}"
            
            day_score = 0
            if use_recommendations:
                subject_lower = subject.lower()
                if subject_lower in recommendations.lower():
                    day_score += 2
                
                if "ریاضی" in subject_lower and day in ["شنبه", "سه‌شنبه"]:
                    day_score += 1
                if "زبان" in subject_lower and day in ["یکشنبه", "چهارشنبه"]:
                    day_score += 1
                if ("تاریخ" in subject_lower or "جغرافیا" in subject_lower) and day == "پنج‌شنبه":
                    day_score += 1
            
           
            study_plan_data.append({
                "روز": day,
                "تاریخ": date_str,
                "درس": subject,
                "زمان شروع": start_time,
                "مدت (ساعت)": duration,
                "امتیاز تناسب": day_score,
                "اولویت": priorities[idx]
            })
    

    study_plan_df = pd.DataFrame(study_plan_data)
    study_plan_df = study_plan_df.sort_values(by=["تاریخ", "زمان شروع"])
    
    final_plan = optimize_study_plan(study_plan_df.to_dict('records'), daily_hours)
    
    return final_plan

def optimize_study_plan(study_plan_data, daily_hours):
    days = {}
    for item in study_plan_data:
        day = item["روز"]
        if day not in days:
            days[day] = []
        days[day].append(item)
    
    
    for day, items in days.items():
        total_hours = 0
        for item in items:
            duration = item["مدت (ساعت)"]
            hours, minutes = map(int, duration.split(":"))
            total_hours += hours + minutes / 60
        
        
        if total_hours > daily_hours * 1.2:  
            scale_factor = daily_hours / total_hours
            for item in items:
                duration = item["مدت (ساعت)"]
                hours, minutes = map(int, duration.split(":"))
                duration_hours = hours + minutes / 60
                new_duration_hours = duration_hours * scale_factor
                
                
                new_hours = int(new_duration_hours)
                new_minutes = int((new_duration_hours - new_hours) * 60)
                item["مدت (ساعت)"] = f"{new_hours}:{new_minutes:02d}"
    
    
    optimized_plan = []
    for day_items in days.values():
        optimized_plan.extend(day_items)
    
    optimized_plan.sort(key=lambda x: (x["تاریخ"], x["زمان شروع"]))
    
    return optimized_plan