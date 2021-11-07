Create Table Deaths_by_Month(
month varchar,
month_code date,
deaths int);

Create Table Deaths_by_Race(
race varchar,
race_code varchar PRIMARY KEY,
deaths int,
population int,
crude_rate float);

Create Table Deaths_by_State_by_Race(
state varchar, 
state_code int,
race varchar, 
race_code varchar,
deaths int,
FOREIGN KEY(race_code) REFERENCES Deaths_by_Race(race_code)
);

Create Table Gender_Deaths(
gender varchar,
gender_code varchar PRIMARY KEY,
deaths int,
population int,
crude_rate float);

Create Table Deaths_by_State_by_Gender(
state varchar,           
state_code int,
gender varchar,
gender_code varchar,
deaths int,
population int,
crude_rate float,
FOREIGN KEY(gender_code) REFERENCES Gender_Deaths(gender_code)
);

Create Table Deaths_by_Age_Group(
five_year_age_groups varchar,
five_year_age_groups_code varchar PRIMARY KEY,
deaths int
);


Create Table Deaths_by_State_by_Age_Group(
state varchar,
state_code int,
five_year_age_groups varchar,
five_year_age_groups_code varchar,
deaths int,
FOREIGN KEY(five_year_age_groups_code) REFERENCES Deaths_by_Age_Group(five_year_age_groups_code)
);




