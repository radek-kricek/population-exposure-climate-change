# How does climate change feel around the globe?
Exposure of populations across the world to extreme heat. Final project at [Spiced Academy](https://www.spiced-academy.com/en). Coded with Python.

Final presentation can be found in 'presentation_graduation.pdf'.


## Questions

1. What percentage of people has direct experience with extreme heat?
2. Does this number change over time?
3. How much is it related to the global temperature anomaly?
4. Is there a clear link between wealth and heat exposure of populations?


## Data

I obtained most of the data from public database of the Organisation for Economic Co-operation and Development (OECD). Specifically, data set on [population exposures of individual countries to different teperature phenomena](https://data-explorer.oecd.org/vis?fs[0]=Topic%2C1%7CEnvironment%23ENV%23%7CAir%20and%20climate%23ENV_AC%23&pg=0&fc=Topic&bp=true&snb=7&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_ECH%40EXT_TEMP&df[ag]=OECD.ENV.EPI&df[vs]=1.1&pd=%2C&dq=AFG%2BBFA%2BBDI%2BCAF%2BTCD%2BSWZ%2BETH%2BGUF%2BATF%2BGLP%2BHMD%2BVAT%2BKAZ%2BXKV%2BKGZ%2BLAO%2BLSO%2BLIE%2BMWI%2BMLI%2BMTQ%2BMDA%2BMNG%2BNPL%2BNER%2BNFK%2BMKD%2BPRY%2BREU%2BRWA%2BSMR%2BSRB%2BSGS%2BSSD%2BTJK%2BTKM%2BUGA%2BUMI%2BUZB%2BZMB%2BZWE%2BIOT%2BBVT%2BBWA%2BBOL%2BBTN%2BBLR%2BAZE%2BARM%2BAND%2BCHE%2BSVK%2BLUX%2BHUN%2BCZE%2BAUT%2BALB%2BDZA%2BASM%2BAGO%2BAIA%2BATG%2BARG%2BABW%2BBHS%2BBHR%2BGGY%2BBGD%2BBRB%2BBLZ%2BBEN%2BBMU%2BBIH%2BBRA%2BVGB%2BBRN%2BBGR%2BCPV%2BKHM%2BCMR%2BCYM%2BCHN%2BCXR%2BCOM%2BCOG%2BCOK%2BCIV%2BHRV%2BCUB%2BCYP%2BPRK%2BCOD%2BDJI%2BDMA%2BDOM%2BECU%2BEGY%2BSLV%2BGNQ%2BERI%2BFRO%2BFJI%2BPYF%2BGAB%2BGMB%2BGEO%2BGHA%2BGRL%2BGRD%2BGUM%2BGTM%2BGIN%2BGNB%2BGUY%2BHTI%2BHND%2BHKG%2BIND%2BIDN%2BIRN%2BIRQ%2BIMN%2BJAM%2BJEY%2BJOR%2BKEN%2BKIR%2BKWT%2BLBN%2BLBR%2BLBY%2BMAC%2BMDG%2BMYS%2BMDV%2BMLT%2BMHL%2BMRT%2BMUS%2BMYT%2BFSM%2BMCO%2BMNE%2BMSR%2BMAR%2BMOZ%2BMMR%2BNAM%2BNRU%2BNCL%2BNIC%2BNGA%2BNIU%2BMNP%2BOMN%2BPAK%2BPLW%2BPSE%2BPAN%2BPNG%2BPER%2BPHL%2BPCN%2BPRI%2BQAT%2BROU%2BRUS%2BSHN%2BKNA%2BLCA%2BSPM%2BVCT%2BWSM%2BSTP%2BSAU%2BSEN%2BSYC%2BSLE%2BSGP%2BSLB%2BSOM%2BZAF%2BLKA%2BSDN%2BSUR%2BSJM%2BSYR%2BTWN%2BTZA%2BTHA%2BTLS%2BTGO%2BTKL%2BTON%2BTTO%2BTUN%2BTCA%2BTUV%2BUKR%2BARE%2BVIR%2BURY%2BVUT%2BVEN%2BVNM%2BWLF%2BESH%2BYEM%2BAU1%2BAU2%2BAU3%2BAU4%2BAU5%2BAU6%2BAU7%2BAU8%2BAUS%2BBEL%2BCAN%2BCHL%2BCOL%2BCRI%2BDNK%2BEST%2BFIN%2BFRA%2BDEU%2BGRC%2BISL%2BIRL%2BISR%2BITA%2BJPN%2BKOR%2BLVA%2BLTU%2BMEX%2BNLD%2BNZL%2BNOR%2BPOL%2BPRT%2BSVN%2BESP%2BSWE%2BTUR%2BGBR%2BUSA%2BG7%2BG20%2BEA19%2BEU27%2BOECD%2BOECDA%2BOECDSO%2BOECDE%2BAES%2BEMES%2BIPAC.A.HD_POP_EXP..W_LT_2...&ly[rw]=REF_AREA&ly[cl]=TIME_PERIOD&to[TIME_PERIOD]=false&lo=5&lom=LASTNPERIODS), and [historical populations of countries](https://data-explorer.oecd.org/vis?fs[0]=Topic%2C1%7CSociety%23SOC%23%7CDemography%23SOC_DEM%23&pg=0&fc=Topic&bp=true&snb=2&df[ds]=dsDisseminateFinalDMZ&df[id]=DSD_POPULATION%40DF_POP_HIST&df[ag]=OECD.ELS.SAE&df[vs]=1.0&pd=2010%2C2021&dq=AUS..PS._T..&ly[rw]=AGE&ly[cl]=TIME_PERIOD&to[TIME_PERIOD]=false).

In addition, I used [NASA data on temperature anomaly](https://data.giss.nasa.gov/gistemp/) and [UN data on GDP per capita](http://data.un.org/Data.aspx?q=gdp+per+capita&d=SNAAMA&f=grID%3a101%3bcurrID%3aUSD%3bpcFlag%3a1).

All for datasets can be found in the 'data' folder as CSV files in its raw form.


## Cleaning and wrangling with Pandas and Pycountry

The whole cleaning process is described in the cleaning.ipynb notebook in the 'notebooks' folder.

Some examples of used techniques include filtering for relevant rows and columns. For example, unnecessary rows with data about provinces can be distinguished by containing numerical characters in the 'ref_area' column, for which reason they were dropped.
```
# only 'ref_area' without numerical characters
df_exp = df_exp.loc[~(df_exp['ref_area'].str.endswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')))]
df_exp.reset_index(drop=True, inplace=True)

# only relevant values of 'measure', 'sex', 'age'
df_pop = df_pop[(df_pop['measure']=='POP') & (df_pop['unit_measure']=='PS') & (df_pop['sex']=='_T') &\
                (df_pop['age']=='_T')]
df_pop.reset_index(drop=True, inplace=True)
df_pop = df_pop.drop(['measure', 'unit_measure', 'sex', 'age'], axis='columns')
```

Creating new columns with country names using alpha3 codes and vice versa.

```
# alpha3 to country
country_alpha3 = {}
for country in pycountry.countries:
    country_alpha3[country.alpha_3] = country.name

# apply
df_exp['country'] = df_exp['ref_area'].map(country_alpha3)   # new column with country names
df_exp = df_exp[['ref_area', 'country', 'measure', 'duration', 'time_period', 'obs_value']]   # reorder columns

# country to alpha3
country_alpha3_inversed = {}
for country in pycountry.countries:
    country_alpha3_inversed[country.name] = country.alpha_3

# change keys for countries with different names in UN data compared to Pycountries
old_keys = ['China', 'United Kingdom', 'Turkey', "Korea, Democratic People's Republic of", "Korea, Republic of",\
           'Venezuela, Bolivarian Republic of', 'Bolivia, Plurinational State of',\
            'Congo, The Democratic Republic of the', 'Tanzania, United Republic of', 'Moldova, Republic of',\
            'North Macedonia']
new_keys = ['China (mainland)', 'United Kingdom of Great Britain and Northern Ireland', 'Türkiye',\
           "Democratic People's Republic of Korea", "Republic of Korea", 'Venezuela (Bolivarian Republic of)',\
           'Bolivia (Plurinational State of)', 'Democratic Republic of the Congo',\
            'United Republic of Tanzania: Mainland', 'Republic of Moldova', 'Republic of North Macedonia']
for key in range(len(new_keys)):
    old_key = old_keys[key]
    new_key = new_keys[key]
    country_alpha3_inversed[new_key] = country_alpha3_inversed.pop(old_key)

# apply
df_gdp['ref_area'] = df_gdp['country'].map(country_alpha3_inversed)
df_gdp = df_gdp[['ref_area', 'country', 'time_period', 'gdp']]   # redorder columns
```

Creating a column with average values.

```
# relevant years in climate data
df_temp_anomaly = df_temp_anomaly[df_temp_anomaly['year'].between(1979,2021)]
df_temp_anomaly = df_temp_anomaly.reset_index(drop=True)

# change data type to perform calculations
df_temp_anomaly[df_temp_anomaly.columns[1:13]] = df_temp_anomaly[df_temp_anomaly.columns[1:13]].astype(float)

# keep only years and yearly averages
df_temp_anomaly['avg_anomaly'] = df_temp_anomaly.iloc[:, 1:].mean(axis=1)   # average over months in a new column
df_temp_anomaly = df_temp_anomaly[['year', 'avg_anomaly']]
```


## Analysis with Pandas, Scikit-learn and SciPy

The whole analysis is described in the analysis.ipynb notebook in the 'notebooks' folder.

Large number of analytical steps was performed with the cleaned data. These steps included for example:

Calculating correlation coefficients.

```
countries = list(df_analyze_corr['ref_area'].unique())   # list of countries present
corr_coeffs = []   # empty list for correlation coefficients for each country

for country in countries:
    df_country = df_analyze_corr[df_analyze_corr['ref_area']==country]   # filter for the country
    df_corr = df_country.corr(numeric_only=True)   # calculate correlation coefficients for numeric columns
    coeff = df_corr.iloc[0,1]   # choose the relevant number
    corr_coeffs.append(coeff)   # store the coefficient inside of the list
```

Calculating weighted averages.

```
years = list(df_exp_pop['time_period'].unique())   # list of years present
exp_world = []   # empty list for worldwide exposures

for year in years:
    df_calc = df_exp_pop[df_exp_pop['time_period']==year]   # filter for the year
    avg_world = sum(df_calc['exposure'] * df_calc['population'])/ sum(df_calc['population'])   # weighted average
    exp_world.append(avg_world)   # store the average inside of the list
```

Merging data frames, grouping and clustering with k-means.

```
# merge
df_exp_gdp = pd.merge(df_exp_summed, df_gdp, how='inner', on=['ref_area', 'time_period'])

# group by and calculate averages
df_exp_gdp_avg = df_exp_gdp.groupby('ref_area')[['exposure', 'gdp']].mean()

# standardize data
scaler = StandardScaler()
scaler.fit(df_num)
df_num_scaled = scaler.transform(df_num) # array of standardized data
df_standardized = pd.DataFrame(df_num_scaled, columns=df_num.columns) # data frame from the array

# fit and assign to clusters
kmeans = KMeans(n_clusters=2) # choose number of clusters
clusters = kmeans.predict(df_standardized)
kmeans.fit(df_standardized)
```

A/B testing

```
df_cluster_0 = df_clustered[df_clustered['cluster']=='0']
df_cluster_1 = df_clustered[df_clustered['cluster']=='1']

test_statistic, pvalue = stats.ttest_ind(df_cluster_0['gdp'], df_cluster_1['gdp'])
print ('GDP:', test_statistic, pvalue)

test_statistic, pvalue = stats.ttest_ind(df_cluster_0['exposure'], df_cluster_1['exposure'])
print ('Exposure:', test_statistic, pvalue)
```


## Visualizations with Plotly

Making visualizations and animations is part of the vizzes.ipynb notebook in the 'notebooks' folder.

One example is shown here, demonstrating making an interactive visualization showing choropleth map.

```
# choropleth map
fig = px.choropleth(data_frame=df_plot, 
                    locations='ref_area', 
                    projection='natural earth',
                    color='color', 
                    locationmode='ISO-3',
                    animation_frame='time_period',   # animating with years
                    animation_group='ref_area',
                    height=600,
                    color_discrete_map=worldmap_colors,
                    title="Rolling Average Across Time Periods",
                    labels={'obs_value': 'Exposure', 'time_period':'Year', 'ref_area':'Country'}   # understandable labels
                   )

# caption showing current year
for frame in fig.frames:
    year = frame['name']
    caption = f'Year: {year}'
    frame['layout'].update(
        annotations=[
            go.layout.Annotation(
                text=caption,
                showarrow=False,
                x=0.5,
                y=0.05,
                xanchor='center',
                yanchor='bottom',
                font=dict(size=18, color="black"),
                bgcolor='white',
                opacity=0.8
            )
        ]
    )

# background colors
fig.update_geos(bgcolor='white')
fig.update_layout(geo=dict(bgcolor='#FFFAE5'))

# improve overall layout
fig.update_layout(
    margin=dict(l=0, r=0, t=100, b=0),   # spaces around map
    title_font=dict(size=24),   # title
    font=dict(family='Arial', size=12, color='black'),   # title
    showlegend=False   # can be set to True to display legend
)

# save animation as html and show
fig.write_html('../vizzes/animation.html')
fig.show()
```
