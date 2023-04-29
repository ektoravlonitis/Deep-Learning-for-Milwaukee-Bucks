# BUCKS STATS

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd
import time

# Get the team ID for the Milwaukee Bucks
bucks_id = teams.find_teams_by_full_name('Milwaukee Bucks')[0]['id']

# Define the seasons you want to get data for
seasons = ['2018-19', '2019-20', '2020-21', '2021-22', '2022-23']

# Create an empty list to store the data for each season
data = [] 

# Loop through each season and retrieve the data for all Bucks games
for season in seasons:
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=bucks_id, season_nullable=season)
    games = gamefinder.get_data_frames()[0]
    data.append(games)
    time.sleep(1)


#win_or_lose = games['WL']

# Combine the data for all seasons into a single DataFrame
bucks_games = pd.concat(data, ignore_index=True)

for i in range(0, len(bucks_games['MATCHUP'])):
    if '@' in bucks_games['MATCHUP'][i]:
        bucks_games['MATCHUP'][i] = 'Away'
    else:
        bucks_games['MATCHUP'][i] = 'Home'

bucks_games = bucks_games.drop(['SEASON_ID', 'TEAM_ID','TEAM_ABBREVIATION', 'TEAM_NAME',
                                'GAME_DATE'
                                ], axis=1)

bucks_games = bucks_games.rename(columns={'PTS': 'PTS_TEAM', 'MIN': 'MIN_TEAM', 'FGM': 'FGM_TEAM',
                                          'FGA': 'FGA_TEAM', 'FG_PCT': 'FG_PCT_TEAM', 'FG3M': 'FG3M_TEAM',
                                          'FG3A': 'FG3A_TEAM', 'FG3_PCT': 'FG3_PCT_TEAM', 'FTM': 'FTM_TEAM',
                                          'FTA': 'FTA_TEAM', 'FT_PCT': 'FT_PCT_TEAM', 'OREB': 'OREB_TEAM',
                                          'DREB': 'DREB_TEAM', 'REB': 'REB_TEAM', 'AST': 'AST_TEAM',
                                          'STL': 'STL_TEAM', 'BLK': 'BLK_TEAM', 'PF': 'PF_TEAM',
                                          'PLUS_MINUS': 'PLUS_MINUS_TEAM', 'TOV': 'TOV_PCT_TEAM',
                                          'MATCHUP': 'HOME/AWAY'})

# Retrieve the player statistics for each game
from nba_api.stats.endpoints import boxscoretraditionalv2
player_stats = []
opposing_team_stats = []

for game_id in bucks_games['GAME_ID']:
    stats = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
    stats_df = stats.get_data_frames()[0]
    # Retrieve the statistics for both teams in the game
    team_stats = stats.get_data_frames()[1]
    
    if team_stats['TEAM_ID'][0] == bucks_id:
        opposing_team_num = 1
    else:
        opposing_team_num = 0
    
    # Append the stats for the opposing team to the opposing_team_stats list
    opposing_team_stats.append(team_stats.iloc[opposing_team_num].to_frame().T)
    
    # Append the stats for the Bucks players to the player_stats list
    player_stats.append(stats_df)

    time.sleep(2)

# Combine the player statistics for all games into a single DataFrame
bucks_stats = pd.concat(player_stats, ignore_index=True)
# Combine the opposing team statistics for all games into a single DataFrame
opposing_team_stats_df = pd.concat(opposing_team_stats, ignore_index=True)
# Filter the DataFrame to only include Bucks players
bucks_roster_stats = bucks_stats[bucks_stats['TEAM_ID'] == bucks_id]
#other_team_stats = bucks_stats[bucks_stats['TEAM_ID'] != bucks_id]
bucks_roster_stats = bucks_roster_stats.drop(['TEAM_ID', 'TEAM_ABBREVIATION','TEAM_CITY', 'PLAYER_ID',
                                              'NICKNAME', 'START_POSITION', 'COMMENT'
                                              ], axis=1)

# Filter the opposing team statistics DataFrame to only include the columns we want
opposing_team_stats_df = opposing_team_stats_df[['GAME_ID', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                                                 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
                                                 'AST', 'STL', 'BLK', 'TO', 'PF']]
# Rename the columns in the opposing team statistics DataFrame to match the column names in the Bucks statistics DataFrame
opposing_team_stats_df = opposing_team_stats_df.rename(columns={'PTS': 'PTS_OPP', 'FGM': 'FGM_OPP', 'FGA': 'FGA_OPP',
                                                                'FG_PCT': 'FG_PCT_OPP', 'FG3M': 'FG3M_OPP', 'FG3A': 'FG3A_OPP',
                                                                'FG3_PCT': 'FG3_PCT_OPP', 'FTM': 'FTM_OPP', 'FTA': 'FTA_OPP',
                                                                'FT_PCT': 'FT_PCT_OPP', 'OREB': 'OREB_OPP', 'DREB': 'DREB_OPP',
                                                                'REB': 'REB_OPP', 'AST': 'AST_OPP', 'STL': 'STL_OPP',
                                                                'BLK': 'BLK_OPP', 'TO': 'TOV_OPP', 'PF': 'PF_OPP'})

players_of_interest = ['Giannis Antetokounmpo', 'Khris Middleton', 'Jrue Holiday',
                       'Brook Lopez', 'Pat Connaughton', 'Bobby Portis',
                       'George Hill', 'Wesley Matthews', 'Jordan Nwora',
                       'Thanasis Antetokounmpo', 'Jevon Carter', 'Serge Ibaka',
                       'Sandro Mamukelashvili', 'Grayson Allen', 'MarJon Beauchamp',
                       'Joe Ingles', 'AJ Green']

bucks_roster_stats = bucks_roster_stats.loc[bucks_roster_stats['PLAYER_NAME'].isin(players_of_interest)]

# create a new dataframe with game-level statistics
new_df = pd.pivot(bucks_roster_stats, index=['GAME_ID'], columns='PLAYER_NAME', values=['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
       'BLK', 'TO', 'PF', 'PTS'])
# flatten the multi-level column index
new_df.columns = ['_'.join(col).rstrip('_') for col in new_df.columns.values]
# reset the index to make game ID and team abbreviation columns
new_df = new_df.reset_index()
merged_df = pd.merge(bucks_games, new_df, on='GAME_ID', how='left')
merged_df2 = pd.merge(merged_df, opposing_team_stats_df, on='GAME_ID', how='left')
merged_df2.to_csv('bucks_roster_stats2.csv', index=False)