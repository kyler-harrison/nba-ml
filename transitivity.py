import numpy as np
import pandas as pd
import time


# TODO if using an algo where you update the matrix as the season progresses,
# the update return is wrong when using mean values, need to change so that
# it just reiterates through the file and gets a new mean (not add the old and
# new matrices like it does now)
# desc.: create/update the matrix of game data, using mean score difference
#        between number of enounters of each matchup
# input: data, in the list format [year, team_0, team_1, ..., outcome]
#        initial_matrix, matrix to which this adds (if not the initial fraction)
#        teams, list of team names
# output: games_matrix, if the first fraction of season
#         initial_matrix + games_matrix, if updating season values
def update_matrix(data, initial_matrix, teams):
    games_matrix = []
    for team in teams:
        team_idx = teams.index(team)
        matchups = [[team, opponent] for opponent in teams]
        team_row = [0] * len(teams)

        for matchup in matchups:
            opponent = matchup[1]  # name of opposing team
            opponent_idx = teams.index(opponent)  # opposing team matrix index
            total_score = 0
            num_encounters = 0
            for game in data:
                if game[1] == team and game[2] == opponent:
                    num_encounters += 1
                    total_score += game[5]  # score difference
                elif game[1] == opponent and game[2] == team:
                    num_encounters += 1
                    total_score += -game[5]  # flip sign, my made up convention

            if num_encounters > 0:
                mean_score_diff = total_score / num_encounters
                team_row[opponent_idx] = mean_score_diff  # mean value into matrix
            else:
                team_row[opponent_idx] = 0

        games_matrix.append(team_row)  # team row is added, moving onto next

    games_matrix = np.array(games_matrix)

    if initial_matrix.size == 0:
        return games_matrix
    else:
        return initial_matrix + games_matrix


# desc.: load game data in from file
# input: file path of data,
#        denominator, initial fraction of season to use (and update)
# output: games_matrix, matrix of the fraction of season
#         remaining_data, rest of season that was not in fraction
#         teams, names of teams in the order of the matrix
def load_data(file_path, denominator):
    year = file_path[5:9]
    file = open(file_path, 'r')
    data = pd.read_csv(file)
    all_data = data.to_numpy()

    initial_data = all_data[:int(all_data.shape[0] / denominator)]
    remaining_data = all_data[int(all_data.shape[0] / denominator):]

    # get unique team names into list, this is in a different order every time
    teams = list(set(data["team_0"].to_numpy()))
    games_matrix = update_matrix(initial_data, np.array([]), teams)

    file.close()

    return games_matrix, remaining_data, teams


# TODO change this function so that it outputs a list of each step's value, no
# input weights
# desc.: the recursive white whale, beached at last
# input: games_matrix, the big matrix of all teams and their outcomes
#        row, row (array) of the team of question (updates with deeper steps)
#        opp, index of opponent in matrix (used to find relationship with team rows)
#        step_lim, maximum number of steps to be computed
#        step_count, starting step (should always inputted inputted as 1)
#        weights, list of weights (one for each step [one step, two step, etc])
def recurse_thy_name(games_matrix, row, opp, step_lim, step_count, weights):
    total = 0
    for i, val in enumerate(row):
        if i == opp and val != 0:
            total += val * weights[step_count - 1]  # value against opp found in row
        else:
            if i != opp and val != 0 and games_matrix[i][opp] != 0 and step_count < step_lim:
                # add this value to the successive step value, multiply by
                # corresponding weight, divide by step_count + 1 to get the mean
                # mean may or may not make some predictions better/worse
                total += (val + recurse_thy_name(games_matrix, games_matrix[i], opp, step_lim, step_count + 1, weights)) * weights[step_count] / (step_count + 1)

    return total


def season_acc(file_path):
    games_matrix, remaining_data, teams = load_data(file_path, 3)

    weights = [1, 0.75, 0.50, 0.25]  # 63.5% accuracy average on score diff
    #weights = [1.278781815514852, 0.8428637208946319, 0.9504861366694738, 0.4712965107223627]
    step_lim = 4
    step_count = 1
    correct = 0
    for game in remaining_data:
        # find the matrix index of each team in this game
        toq = teams.index(game[1])
        opp = teams.index(game[2])
        toq_row = games_matrix[toq]  # row of game values for team in question
        #print("{} vs. {}".format(teams[toq], teams[opp]))

        #prediction = trans_pred(toq, opp, games_matrix, teams)
        prediction = recurse_thy_name(games_matrix, toq_row, opp, step_lim, step_count, weights)
        #prediction = proper1(games_matrix, toq_row, toq, opp, step_lim, step_count, weights)
        #print("Predicted diff.: {}".format(prediction))
        #print("Actual diff.: {}".format(game[5]))

        if game[5] < 0:
            actual_winner = game[2]
        else:
            actual_winner = game[1]

        if prediction < 0:
            pred_winner = game[2]
        else:
            pred_winner = game[1]

        if actual_winner == pred_winner:
            correct += 1

        #print("Predicted winner: {}".format(pred_winner))
        #print("Actual winner: {}\n".format(actual_winner))

    accuracy = correct / len(remaining_data)
    print("Accuracy: {}".format(accuracy))

    return accuracy


def main():
    start = time.time()
    file_names = ["data/" + str(year) + "_games.csv" for year in range(2001, 2020)]
    accuracies = [season_acc(path) for path in file_names]
    print("\n\n\n\n")
    print(accuracies)
    print("Mean acc.: {}".format(sum(accuracies) / len(accuracies)))
    end = time.time()
    print("time (sec): {}".format(end - start))


if __name__ == "__main__":
    main()
