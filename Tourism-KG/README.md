# Chinese Tourism KG

[SPARQL endpoint](http://166.111.68.66:8891/sparql)

Attraction
```SPARQL
select distinct ?c
from <travel7>
where {
?t rdfs:subClassOf* <http://travel.org/景点> .
?s a ?t .
?s <http://travel.org/Name> ?c .

} limit 10000
```

Brand
```SPARQL
select distinct ?c
from <travel7>
where {
?t rdfs:subClassOf* <http://travel.org/品牌> .
?t rdfs:label ?c .
# <http://travel.org/pinpai3> ?p ?o
# <http://travel.org/pinpai3> <http://travel.org/RecommendShop> ?shop
} limit 10000
```

Person
```SPARQL
select distinct ?c
from <travel7>
where {
?s a <http://travel.org/人物> .
?s <http://travel.org/ChineseName> ?c
} limit 100
```

Relic
```SPARQL
select distinct ?c
from <travel7>
where {
?s a <http://travel.org/文物> .
?s <http://travel.org/ChineseName> ?c
} limit 1000
```

Building
```SPARQL
select distinct ?c
from <travel7>
where {
?s a <http://travel.org/建筑> .
?s <http://travel.org/ChineseName> ?c
} limit 1000
```

Organization
```SPARQL
select distinct ?c
from <travel7>
where {
?s a <http://travel.org/组织机构> .
?s <http://travel.org/ChineseName> ?c
} limit 1000
```

Dishes
```SPARQL
select distinct ?c
from <travel7>
where {
?s a <http://travel.org/菜品> .
?s <http://travel.org/Name> ?c
} limit 1000
```

union

Cannot use multiple transitions under multiple unions

Retrieve up to 10,000 results each time
```SPARQL
select distinct ?c
from <travel7>
where{
{?t rdfs:subClassOf* <http://travel.org/景点> .
?s a ?t .
?s <http://travel.org/Name> ?c .
}
union
{?t rdfs:subClassOf* <http://travel.org/品牌> .
?t rdfs:label ?c .}
union
{?s a <http://travel.org/人物> .
?s <http://travel.org/ChineseName> ?c}
union
{?s a <http://travel.org/文物> .
?s <http://travel.org/ChineseName> ?c}
union
{?s a <http://travel.org/建筑> .
?s <http://travel.org/ChineseName> ?c}
union
{?s a <http://travel.org/组织机构> .
?s <http://travel.org/ChineseName> ?c}
union
{?s a <http://travel.org/菜品> .
?s <http://travel.org/Name> ?c}
}
```

Attributes of each entity
```SPARQL
select distinct ?s ?p ?o
from <travel7>
where {
?t rdfs:subClassOf* <http://travel.org/景点> .
?s a ?t .
?s ?p ?o .
filter (?p != <http://travel.org/ImageLink>)
filter (?p != <http://travel.org/BestTravelTime>)
filter (?p != <http://travel.org/StartHoursOfMorningOnWeekends>)
filter (?p != <http://travel.org/EndHoursOfMorningOnWeekdays>)
filter (?p != <http://travel.org/OpeningDays>)
} 
# order by asc(?s)
limit 10000
```

```SPARQL
select distinct ?s ?o ?p ?os
from <travel7>
where {
?t rdfs:subClassOf* <http://travel.org/品牌> .
?s a ?t .
?s <http://travel.org/RecommendShop> ?o .
?o ?p ?os .
filter (?p != <http://travel.org/ImageLink>)
filter (?p != <http://travel.org/BestTravelTime>)
filter (?p != <http://travel.org/StartHoursOfMorningOnWeekends>)
filter (?p != <http://travel.org/EndHoursOfMorningOnWeekdays>)
filter (?p != <http://travel.org/OpeningDays>)
} 
limit 10000 offset 0
```

```SPARQL
select count(?s) as ?c
from <travel7>
where {
?s a ?t .
?s ?p ?o .
filter regex(?t, "门店")
}
```