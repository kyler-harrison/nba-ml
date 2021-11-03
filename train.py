import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from tensorflow.keras.models import load_model
from pickle import dump
from pickle import load
import statistics as stat


# TODO 
# Dear future Ky, 
# I hope you are more clean and organized than this. I know I should have broke many of these things down multiple files, 
# but I just had too many ideas and things got out of hand. 
# See you in hell,
# Ky (8/14/2020)

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


# desc.: the transitive algo without the step_occurences (mean values for each step value)
def transitive_recursion_algo_no_mean(games_matrix, row, row_idx, opp_idx, step_lim, step_count):
    total = np.zeros(step_lim)
    for i, val in enumerate(row):
        if i == opp_idx and val != 0:
            total[step_count - 1] += val
        elif i != opp_idx and i != row_idx and val != 0 and games_matrix[i][opp_idx] != 0 and step_count < step_lim:
            total[step_count] += val
            total += transitive_recursion_algo_no_mean(games_matrix, games_matrix[i], row_idx, opp_idx, step_lim, step_count + 1)
    return total


# desc.: the main algorithm that uses the mean score differences of games to
#        values for each step, need to divide the total array by the
#        step_occurences array (after any zeros have been changed to ones) to
#        get a final array containing the mean values for each step
# input: games_matrix, matrix of all the game data
#        row, row (array) of the team of question
#        row_idx, index of the team of question row
#        opp_idx, index of the opponent row
#        step_lim, max number of steps to take
#        step_count, starting step number (alwasy input as 1)
#        step_occurences, should always be a 1 x step_lim array of zeros
# output: total, array of total values for each step / step_number (sum of all
#                of that step_number)
#         step_occurences, number of relationships found for each step_number
def transitive_recursion_algo(games_matrix, row, row_idx, opp_idx, step_lim, step_count, step_occurences):
    total = np.zeros(step_lim)
    for i, val in enumerate(row):
        if i == opp_idx and val != 0:
            total[step_count - 1] += val
            step_occurences[step_count - 1] += 1
        elif i != opp_idx and i != row_idx and val != 0 and games_matrix[i][opp_idx] != 0 and step_count < step_lim:
            total[step_count] += val
            total += transitive_recursion_algo(games_matrix, games_matrix[i], row_idx, opp_idx, step_lim, step_count + 1, step_occurences)[0]
    return total, step_occurences


# desc.: create a csv of the data returned from the recursive function for each team
#        in each game, ex. following line
#        team_0 step 0 val, ..., team_0 step n val, team_1 step 0 val, ..., team_1 step n val, outcome (+1/-1 or 0/1)
# input: training_path, path for the new data to be stored
#        season_paths, list of all seasons (containing game data)
#        season_denom, denominator of the season fraction
#        step_count, min number of step relationships to evaluate
#        step_lim, max number of step relationships to evaluate
def assemble_training_file(training_path, season_paths, season_denom, step_count, step_lim):
    training_file = open(training_path, "w")
    # it just works
    header = ','.join(["team_0_step_" + str(num) for num in range(step_lim)] + ["team_1_step_" + str(num) for num in range(step_lim)]) + ",outcome\n"
    training_file.write(header)

    for season_path in season_paths:
        start = time.time()
        season_file = open(season_path, "r")
        games_matrix, remaining_data, teams = load_data(season_path, season_denom)

        for game in remaining_data:
            toq_idx = teams.index(game[1])  # index of team_0
            opp_idx = teams.index(game[2])  # index of team_1
            toq_row = games_matrix[toq_idx]  # array of mean game data for team_0
            opp_row = games_matrix[opp_idx]  # array of mean game data for team_1

            # two of these that reset for each game because arrays change when
            # passed into functions
            step_occurences_0 = np.zeros(step_lim)
            step_occurences_1 = np.zeros(step_lim)

            # get the array of step values for each team
            team_0_step_vals_arr, team_0_step_occurences = transitive_recursion_algo(games_matrix, toq_row, toq_idx, opp_idx, step_lim, step_count, step_occurences_0)
            team_1_step_vals_arr, team_1_step_occurences = transitive_recursion_algo(games_matrix, opp_row, opp_idx, toq_idx, step_lim, step_count, step_occurences_1)

            for i, value in enumerate(team_0_step_occurences):
                if team_0_step_occurences[i] == 0:
                    team_0_step_occurences[i] = 1
                if team_1_step_occurences[i] == 0:
                    team_1_step_occurences[i] = 1

            # the sacred values
            team_0_final = team_0_step_vals_arr / team_0_step_occurences
            team_1_final = team_1_step_vals_arr / team_1_step_occurences

            # this is probably inefficient but whatever
            team_0_final_ls = [str(val) for val in team_0_final]
            team_1_final_ls = [str(val) for val in team_1_final]
            combined_vals = team_0_final_ls + team_1_final_ls
            # get the outcome of game (+1/-1 might want to change to 0/1 idk)
            outcome = str(game[6])

            # line to write
            line = ','.join(combined_vals) + ',' + outcome + '\n'
            # write it
            training_file.write(line)

        season_file.close()
        end = time.time()
        print("Season: {}\nTime (sec): {}".format(season_path[-14:9], end - start))
    training_file.close()


# input: start_year, first year of data collect
#        stop_year, the last year of data to collect
#        file_path, file to write data to 
# this is good to create training file (change path in function call first and delete dummies in assemble function) once
# that is fixed, then set up the load in training_file and split x/y, then
# attempt models
def assemble_all_data(start_year, stop_year, file_path):
    season_denom = 3
    step_count = 1
    step_lim = 4
    # saving some validation data for later
    season_paths = ["data/" + str(year) + "_games.csv" for year in range(start_year, stop_year + 1)]

    start = time.time()
    assemble_training_file(file_path, season_paths, season_denom, step_count, step_lim)
    end = time.time()
    print("Time (sec): {}".format(end - start))


# desc.: use to check the recursion function values (are they reasonable?)
def test_recurse():
    path = "data/2004_games.csv"
    games_matrix, remaining_data, teams = load_data(path, 3)
    step_lim = 3
    step_count = 1

    count = 0
    # games_counted = 0
    # correct = 0
    for game in remaining_data:
        toq_idx = teams.index(game[1])  # index of team_0
        opp_idx = teams.index(game[2])  # index of team_1
        toq_row = games_matrix[toq_idx]  # array of mean game data for team_0
        opp_row = games_matrix[opp_idx]  # array of mean game data for team_1

        step_occurences = np.zeros(step_lim)
        step_occurences1 = np.zeros(step_lim)
        team_0_step_vals_arr, counts_0 = transitive_recursion_algo(games_matrix, toq_row, toq_idx, opp_idx, step_lim, step_count, step_occurences)
        team_1_step_vals_arr, counts_1 = transitive_recursion_algo(games_matrix, opp_row, opp_idx, toq_idx, step_lim, step_count, step_occurences1)
        #counts_0 = np.array([count + 1 for count in counts_0 if count == 0])
        for i, value in enumerate(counts_0):
            if counts_0[i] == 0:
                counts_0[i] = 1
            if counts_1[i] == 0:
                counts_1[i] = 1

        # the values that we've done all this work for
        team_0_final = team_0_step_vals_arr / counts_0
        team_1_final = team_1_step_vals_arr / counts_1

        print("Game #{}".format(count))
        # print("team_0 counts: {}".format(counts_0))
        # print("team_1 counts: {}".format(counts_1))
        print("team_0: {}".format(team_0_final))
        print("team_1: {}".format(team_1_final))
        # print('\n')

        # team_0_total = sum([num for num in team_0_final if num > 0])
        # team_1_total = sum([num for num in team_1_final if num > 0])
        # team_0_total = sum(team_0_final)
        # team_1_total = sum(team_1_final)
        # if team_0_total == team_1_total:
        #     winner = 0
        # elif team_0_total > team_1_total:
        #     winner = 1
        # elif team_0_total < team_1_total:
        #     winner = -1
        #
        # if winner != 0:
        #     games_counted += 1
        #     actual_winner = game[6]
        #     if actual_winner == winner:
        #         correct += 1

        count += 1

    # print("Games counted {} / {}:".format(games_counted, count))
    # print("Accuracy: {}".format(correct / games_counted))


def load_and_split(file_path, start, end, scaler_file):
    data = pd.read_csv(file_path)
    x = np.array(data.loc[:, start:end])
    y = np.array(data["outcome"].tolist())

    # classes need to be 0, 1 (they are 1/-1 in the file)
    for i, y_val in enumerate(y):
        if y_val == 1:
            y[i] = 0
        else:
            y[i] = 1

    # standardization (I think)
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)
    dump(scaler, open(scaler_file, "wb"))

    # split into train/test
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

    # for neural network
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    return x_train, x_test, y_train, y_test

# best so far:
# 66.7% acc using input to output layers only (v0)
# 66.4% acc using input to 16 to output (v1)
# 66.3% acc using input-16-16-output (v2)
def nn_model(x_train, x_test, y_train, y_test, model_file_path):
    model = Sequential()
    model.add(Flatten(input_shape=x_train.shape[1:]))

    model.add(Dense(2, activation="softmax"))

    opt = Adam(lr=1e-3, decay=1e-5)
    model.compile(optimizer=opt,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    print(model.summary())
    model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
    loss, acc = model.evaluate(x_test, y_test)
    print("Loss: {}, Accuracy: {}".format(loss, acc))
    model.save(model_file_path)

    return model


def is_same_class(predicted, actual):
    if predicted == actual:
        return 1
    else:
        return 0


def test_nn_model(model_path, season_paths, exists, predictions_path, season_denom, step_count, step_lim, scaler, start, end):
    model = load_model(model_path)  # load in nn model

    # create a file of the team_0 step vals, team_1 step vals, outcome if not already existing
    if not exists:
        assemble_training_file(predictions_path, season_paths, season_denom, step_count, step_lim)

    # open the file, use model on each, output probabilities
    data = pd.read_csv(predictions_path)
    x = np.array(data.loc[:, start:end])
    outcomes = np.array(data["outcome"].tolist())

    # classes need to be 0, 1 (they are 1/-1 in the file)
    for i, outcome_val in enumerate(outcomes):
        if outcome_val == 1:
            outcomes[i] = 0
        else:
            outcomes[i] = 1

    # standardization with the same used on the model data (I think)
    x = scaler.transform(x)
    x = x.reshape(x.shape[0], 1, x.shape[1])
    predictions = model.predict(x)

    # ignore me, come back if something fails
    # predictions_file = open(predictions_path, "r")
    # all_values = []
    # outcomes = []
    # for i, line in enumerate(predictions_file):
    #     if i != 0:
    #         values_ls = line.split(',')
    #         outcome_num = int(values_ls[-1][:-1])
    #         if outcome_num == 1:
    #             outcomes.append(0)  # grab the actual outcome
    #         else:
    #             outcomes.append(1)
    #         values_ls.pop()  # remove outcome, leaving team_0 and team_1 step values
    #         values_arr = np.array([float(val) for val in values_ls])
    #         values_arr = values_arr.reshape(1, values_arr.size)  # reshape for model later
    #         all_values.append(values_arr)
    #         # prediction = model.predict(values_arr.reshape(1, values_arr.shape[0], 1))
    #         # print(prediction)

    # all_values = np.array(all_values)  # this is probably a bad way of doing this but whatever
    # predictions = model.predict(all_values)

    six_sev = 0
    sev_eight = 0
    eight_nine = 0
    nine_hun = 0
    below_six = 0
    below_six_correct = 0
    six_correct = 0
    sev_correct = 0
    eight_correct = 0
    nine_correct = 0
    ovr_correct = 0
    for j, prediction in enumerate(predictions):
        winner_prob = max(prediction)
        winner_class = np.argmax(prediction)
        ovr_correct += is_same_class(winner_class, outcomes[j])

        if winner_prob >= 0.6 and winner_prob < 0.7:
            six_sev += 1
            six_correct += is_same_class(winner_class, outcomes[j])
        elif winner_prob >= 0.7 and winner_prob < 0.8:
            sev_eight += 1
            sev_correct += is_same_class(winner_class, outcomes[j])
        elif winner_prob >= 0.8 and winner_prob < 0.9:
            eight_nine += 1
            eight_correct += is_same_class(winner_class, outcomes[j])
        elif winner_prob >= 0.9 and winner_prob < 1:
            nine_hun += 1
            nine_correct += is_same_class(winner_class, outcomes[j])
        else:
            below_six += 1
            below_six_correct += is_same_class(winner_class, outcomes[j])

    below_six_acc = below_six_correct / below_six
    six_acc = six_correct / six_sev
    sev_acc = sev_correct / sev_eight
    eight_acc = eight_correct / eight_nine
    nine_acc = nine_correct / nine_hun
    total = j + 1
    ovr_accuracy = ovr_correct / total
    
    print("{}\n".format(predictions_path))
    print("overall accuracy: {}".format(ovr_accuracy))
    print("accuracy when prediction confidence < 0.60: correct {} / {}, {}".format(below_six_correct, below_six, below_six_acc))
    print("accuracy when prediction confidence 0.70 >= 0.60: correct {} / {}, {}".format(six_correct, six_sev, six_acc))
    print("accuracy when prediction confidence 0.80 >= 0.70: correct {} / {}, {}".format(sev_correct, sev_eight, sev_acc))
    print("accuracy when prediction confidence 0.90 >= 0.80: correct {} / {}, {}".format(eight_correct, eight_nine, eight_acc))
    print("accuracy when prediction confidence >= 0.90: correct {} / {}, {}\n".format(nine_correct, nine_hun, nine_acc))
    print("total predictions made: {}".format(total))
    print("following is % of model confidence")
    print("below 60%: {}".format(below_six / total))
    print("60-70%: {}".format(six_sev / total))
    print("70-80%: {}".format(sev_eight / total))
    print("80-90%: {}".format(eight_nine / total))
    print("90-100%: {}\n\n".format(nine_hun / total)) 

    segment_corrects = [below_six_correct, six_correct, sev_correct, eight_correct, nine_correct]
    segment_totals = [below_six, six_sev, sev_eight, eight_nine, nine_hun]

    return ovr_accuracy, segment_corrects, segment_totals


# no bueno
def knn_classifier(x_train, x_test, y_train, y_test, neighbors):
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print("Accuracy: {}".format(acc))


# best acc: 66.6% with rbf kernel
def svm_classifier(x_train, x_test, y_train, y_test, kernel):
    classifier = svm.SVC(kernel=kernel, C=2)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: {}".format(accuracy))


# best acc: 66%
def naive_bayes_classifier(x_train, x_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: {}".format(accuracy))


# no bueno
def decision_tree_classifier(x_train, x_test, y_train, y_test):
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    y_pred = dtc.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: {}".format(accuracy))


def main():
    x_train, x_test, y_train, y_test = load_and_split("data/2001_2010_training_data.csv", "team_0_step_0", "team_1_step_3", "scalers/scaler_NA.pkl")
    #model = nn_model(x_train, x_test, y_train, y_test, "nn_models/2001_2010_four_step_v4.model")  # v3 on 2001_2010 acc of 67.3%
    #knn_classifier(x_train, x_test, y_train, y_test, 31)
    #svm_classifier(x_train, x_test, y_train, y_test, "rbf")
    #naive_bayes_classifier(x_train, x_test, y_train, y_test)
    #decision_tree_classifier(x_train, x_test, y_train, y_test)
    scaler = load(open("scalers/scaler_v4.pkl", "rb"))
    season_prediction_paths = ["steps_outcome_data/" + str(year) + "_steps_outcome.csv" for year in range(2011, 2020)]
    season_data_paths = ["data/" + str(year) + "_games.csv" for year in range(2011, 2020)]
    acc_ls = []
    all_segment_corrects = []
    all_segment_totals = []
    for i, season_prediction_path in enumerate(season_prediction_paths):
        acc, segment_corrects, segment_totals = test_nn_model("nn_models/2001_2010_four_step_v4.model", [season_data_paths[i]], True, season_prediction_path, 3, 1, 4, scaler, "team_0_step_0", "team_1_step_3")
        acc_ls.append(acc)
        all_segment_corrects.append(segment_corrects)
        all_segment_totals.append(segment_totals)

    combined_segement_corrects = [0] * 5
    combined_segment_totals = [0] * 5
    for j, ls in enumerate(all_segment_corrects):
        for k, val in enumerate(ls):
            combined_segement_corrects[k] += val 
            combined_segment_totals[k] += all_segment_totals[j][k] 

    print(combined_segement_corrects)
    print(combined_segment_totals)
    segment_accuracies = [val / combined_segment_totals[l] for l, val in enumerate(combined_segement_corrects)]
    print("\nmean acc: {}".format(stat.mean(acc_ls)))
    print("accuracies for each segment (< 0.6, 0.6-0.7, 0.7-0.8, 0.9-1.0\n{}".format(segment_accuracies))

if __name__ == "__main__":
    main()
