## Data Overview

This dataset consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The MovieLens dataset has long been used by industry and academic researchers to improve the performance of explicitly-based recommender systems, and now you get to as well!
For this Predict, we'll be using a special version of the MovieLens dataset which has enriched with additional data, and resampled for fair evaluation purposes.

## Source

The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB

## Supplied Files

**genome_scores.csv** - a score mapping the strength between movies and tag-related properties.

**genome_tags.csv** - user assigned tags for genome-related scores

**imdb_data.csv** - Additional movie metadata scraped from IMDB using the links.csv file.

**links.csv** - File providing a mapping between a MovieLens ID and associated IMDB and TMDB IDs.

**sample_submission.csv** - Sample of the submission format for the hackathon.

**tags.csv** - User assigned for the movies within the dataset.

**test.csv** - The test split of the dataset. Contains user and movie IDs with no rating data.

**train.csv** - The training split of the dataset. Contains user and movie IDs with associated rating data.

## Additional Information

The below information is provided directly from the MovieLens dataset description files:

### Ratings Data File Structure (train.csv)

All ratings are contained in the file train.csv. Each line of this file after the header row represents one rating of one movie by one user, and has the following format:

`userId,movieId,rating,timestamp `

For the purposes of this App we will refer to this file as "Ratings"
The lines within this file are ordered first by userId, then, within user, by movieId.
Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).
Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

### Tags Data File Structure (tags.csv)

All tags are contained in the file tags.csv. Each line of this file after the header row represents one tag applied to one movie by one user, and has the following format:
  
`userId,movieId,tag,timestamp`

The lines within this file are ordered first by userId, then, within user, by movieId.   
Tags are user-generated metadata about movies. Each tag is typically a single word or short phrase. The meaning, value, and purpose of a particular tag is determined by each user.
Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970

### Movies Data File Structure (movies.csv)

Movie information is contained in the file movies.csv. Each line of this file after the header row represents one movie, and has the following format:

`movieId,title,genres` 

Movie titles are entered manually or imported from https://www.themoviedb.org/, and include the year of release in parentheses. Errors and inconsistencies may exist in these titles.
Genres are a pipe-separated list, and are selected from the following:

- Action
- Adventure
- Animation
- Children's
- Comedy
- Crime
- Documentary
- Drama
- Fantasy
- Film-Noir
- Horror
- Musical
- Mystery
- Romance
- Sci-Fi
- Thriller
- War
- Western
- (no genres listed)

### Links Data File Structure (links.csv)

Identifiers that can be used to link to other sources of movie data are contained in the file links.csv. Each line of this file after the header row represents one movie, and has the following format:

`movieId,imdbId,tmdbId`

movieId is an identifier for movies used by https://movielens.org. E.g., the movie Toy Story has the link https://movielens.org/movies/1.
imdbId is an identifier for movies used by http://www.imdb.com. E.g., the movie Toy Story has the link http://www.imdb.com/title/tt0114709/.
tmdbId is an identifier for movies used by https://www.themoviedb.org. E.g., the movie Toy Story has the link https://www.themoviedb.org/movie/862.
Use of the resources listed above is subject to the terms of each provider.

### Tag Genome (genome-scores.csv and genome-tags.csv)

As described in this article, the tag genome encodes how strongly movies exhibit particular properties represented by tags (atmospheric, thought-provoking, realistic, etc.). The tag genome was computed using a machine learning algorithm on user-contributed content including tags, ratings, and textual reviews.
The genome is split into two files. The file genome-scores.csv contains movie-tag relevance data in the following format:

`movieId,tagId,relevance`

The second file, genome-tags.csv, provides the tag descriptions for the tag IDs in the genome file, in the following format:

`tagId,tag`
