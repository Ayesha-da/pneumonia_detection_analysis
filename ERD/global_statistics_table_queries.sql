CREATE TABLE Deaths_by_GDP(
country varchar,
country_code varchar,
year int NOT NULL,
deaths float NOT NULL,
GDP float NOT NULL,
population bigint NOT NULL);

CREATE TABLE Deaths_over_70_risk_factors(
country varchar,
country_code varchar,
year int,
no_access_to_handwashing float,
secondhand_smoke float, 
smoking float, 
pollution float);

CREATE TABLE Global_Child_Mortality_by_Country(
country varchar, 
country_code varchar, 
year int,
deaths float);

CREATE TABLE Pneumonia_Mortality_by_Age(
country varchar,
country_code varchar, 
year int,
less_than_five float, 
five_to_fifteen float, 
fifteen_to_fourty_nine float, 
fifty_to_sixty_nine float, 
seventy_or_greater float
);

CREATE TABLE Child_Deaths_Risk_Factors(
country varchar,
country_code varchar,
year int,
ambient_particle_matter_polution float, 
child_underweight float, 
household_pollution float, 
no_access_to_handwashing float, 
secondhand_smoke float, 
vitamin_A_Deficiency float, 
zinc_deficiency float, 
short_gestation_for_birth_weight float, 
non_exclusive_breastfeeding float, 
low_birth_weight_for_gestation float, 
child_wasting float, 
child_stunting float
);