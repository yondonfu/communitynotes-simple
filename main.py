import math
import random
from typing import Callable


def avg(u: list[float]) -> float:
    return sum(u) / len(u)


def dot_product(u: list[float], v: list[float]) -> float:
    return sum(x[0] * x[1] for x in zip(u, v))


def euclidean_norm(u: list[float]) -> float:
    return math.sqrt(sum(x ** 2 for x in u))


def random_float() -> float:
    return random.randrange(1000) / 1000


def predict_note_rating(global_intercept: float, user_intercept: float, note_intercept: float, user_vector: list[float], note_vector: list[float]) -> float:
    return global_intercept + user_intercept + note_intercept + dot_product(user_vector, note_vector)


def compute_loss(ratings: list[list[float]], global_intercept: float, user_intercepts: list[float], note_intercepts: list[float], user_vectors: list[list[float]], note_vectors: list[list[float]]) -> float:
    n_user = len(ratings)
    n_note = len(ratings[0])

    # Regularization values
    # Regularization for intercepts is 5x higher than regularization for factors
    lambda_intercepts = 0.15
    lambda_factors = 0.03

    loss = 0
    for i in range(n_user):
        user_intercept = user_intercepts[i]
        user_vector = user_vectors[i]

        for j in range(n_note):
            note_intercept = note_intercepts[j]
            note_vector = note_vectors[j]

            note_rating_actual = ratings[i][j]
            note_rating_pred = predict_note_rating(
                global_intercept, user_intercept, note_intercept, user_vector, note_vector)

            loss += (note_rating_actual - note_rating_pred) ** 2

    loss += lambda_intercepts * \
        sum(x ** 2 for x in user_intercepts +
            note_intercepts + [global_intercept])
    loss += lambda_factors * \
        sum(euclidean_norm(x) ** 2 for x in user_vectors + note_vectors)

    return loss


def train(ratings: list[list[float]]):
    n_user = len(ratings)
    n_note = len(ratings[0])
    n_factors = 1

    # Init values
    global_intercept = [random_float()]
    user_intercepts = [random_float() for _ in range(n_user)]
    note_intercepts = [random_float() for _ in range(n_note)]
    # The original algo uses one-dimensional factor vectors to avoid overfitting on a small dataset and because
    # additional factors did not appear to add explanatory power.
    # The authors expect to increase the dimensionality of these vectors with additional factors over time.
    user_vectors = [[random_float() for _ in range(n_factors)]
                    for _ in range(n_user)]
    note_vectors = [[random_float() for _ in range(n_factors)]
                    for _ in range(n_note)]

    model_params = [[global_intercept],
                    user_intercepts, note_intercepts, user_vectors, note_vectors]

    loss = compute_loss(ratings, global_intercept[0], user_intercepts,
                        note_intercepts, user_vectors, note_vectors)

    max_epoch = 9**99

    # Naive descent
    for epoch in range(max_epoch):
        if epoch % 50 == 0:
            print("Epoch: {} Loss: {}".format(epoch, loss))

        epoch_delta = max(0.001, 0.1 / (epoch + 1) ** 0.5)

        start_epoch_loss = loss

        for param in model_params:
            for i in range(len(param)):
                for delta in (epoch_delta, -epoch_delta):
                    if type(param[i]) is list:
                        param[i] = [x+delta for x in param[i]]
                    else:
                        param[i] += delta

                    new_loss = compute_loss(
                        ratings, global_intercept[0], user_intercepts, note_intercepts, user_vectors, note_vectors)

                    if loss < new_loss:
                        # Applying delta against param did not result in reduced loss
                        # Rollback delta for this param
                        if type(param[i]) is list:
                            param[i] = [x-delta for x in param[i]]
                        else:
                            param[i] -= delta
                    else:
                        # Applying delta against param resulted in reduced loss
                        # Track current lowest loss
                        loss = new_loss

        if loss == start_epoch_loss:
            print("Finished descent")
            break

    return (global_intercept, user_intercepts, note_intercepts, user_vectors, note_vectors)


def run_training(n_user: int, n_note: int, get_ratings: Callable[[int, int], list[list[float]]], rounds: int):
    note_intercepts_sum = [0 for _ in range(n_note)]

    for i in range(rounds):
        print("ROUND {}".format(i))

        ratings = get_ratings(n_user, n_note)
        print("Generated ratings")
        for rating in ratings:
            print(rating)

        global_intercept, user_intercepts, note_intercepts, user_vectors, note_vectors = train(
            ratings)

        print("TRAINED MODEL PARAMS")
        print("Global intercept")
        print(global_intercept)
        print("User intercepts")
        print(user_intercepts)
        print("Note intercepts")
        print(note_intercepts)
        print("User vectors")
        for user_vector in user_vectors:
            print(user_vector)
        print("Note vectors")
        for note_vector in note_vectors:
            print(note_vector)

        note_intercepts_sum = [x[0] + x[1]
                               for x in zip(note_intercepts_sum, note_intercepts)]

    note_intercepts_avg = [x / rounds for x in note_intercepts_sum]

    return note_intercepts_avg


def generate_ratings_random(n_user: int, n_note: int, rating_prob: float) -> list[list[float]]:
    # 1.0 = Yes
    # 0.5 = Somewhat
    # 0.0 = N/A
    # -1.0 = No
    # https://communitynotes.twitter.com/guide/en/under-the-hood/ranking-notes uses No = 0.0
    # We use No = -1.0 instead so we can use 0.0 when there is no rating available
    choices = [1.0, 0.5, -1.0]

    ratings = []
    for _ in range(n_user):
        user_rating = []

        for _ in range(n_note):
            if random_float() < rating_prob:
                user_rating.append(choices[random.randrange(len(choices))])
            else:
                user_rating.append(0.0)

        ratings.append(user_rating)

    return ratings


def generate_ratings_polarized(n_user: int, n_note: int, rating_count: int) -> list[list[float]]:
    ratings = [[0.0 for _ in range(n_note)] for _ in range(n_user)]

    perms = []
    # Each permutation maps the user at index i with a value v which identifies a note rated by the user
    while len(perms) < rating_count:
        new_perm = [random.randrange(n_note) for _ in range(n_user)]

        good = True
        for i in range(n_user):
            for existing_perm in perms:
                if existing_perm[i] == new_perm[i]:
                    good = False

        if good:
            perms.append(new_perm)

    print(perms)

    # The top n_user / 2 users are in tribe A
    # The left n_note / 2 notes are in tribe A
    # The bottom n_user / 2 users are in tribe B
    # The right n_note / 2 notes are in tribe B
    # A user rates a note with 1.0 if the note is in the same tribe and -1.0 if the note is in the other tribe
    for perm in perms:
        for i, v in enumerate(perm):
            # User i rated note v
            # Check if user i is in the same tribe as note v
            if (i * 2) // n_user == (v * 2) // n_note:
                ratings[i][v] = 1.0
            else:
                ratings[i][v] = -1.0

    return ratings


def generate_ratings_polarized_bridged(n_user: int, n_note: int, rating_count: int) -> list[list[float]]:
    ratings = generate_ratings_polarized(n_user, n_note, rating_count)

    # At this point, the ratings matrix *only* contains ratings that reflect a user's tribal viewpoint.
    # As is, we would expect the note intercepts computed during training to largely be negative if the model is designed
    # to compute higher note intercepts for notes with ratings from users with *both* tribal viewpoints.

    n_user = len(ratings)
    n_note = len(ratings[0])
    for i in range(n_user):
        # Group 1
        # Notes in this group are helpful and bridge the tribes so flip all existing ratings to 1.0
        for j in range(n_note // 3):
            if ratings[i][j] != 0.0:
                ratings[i][j] = 1.0

        # Group 2
        # Notes in this group are helpful and polarizing so leave existing ratings that already reflect tribal viewpoints

        # Group 3
        # Notes in this group are not helpful so flip all existing ratings to -1.0
        for j in range(n_note - n_note // 3, n_note):
            if ratings[i][j] != 0.0:
                ratings[i][j] = -1.0

    return ratings


if __name__ == "__main__":
    rounds = 1
    n_user = 16
    n_note = 16

    def get_ratings(n_user: int, n_note: int):
        # rating_prob = 0.1
        # return generate_ratings_random(n_user, n_note, rating_prob)
        rating_count = 4
        return generate_ratings_polarized_bridged(n_user, n_note, rating_count)

    note_intercepts = run_training(n_user, n_note, get_ratings, rounds)
    print("Note intercepts (average over {} rounds)".format(rounds))
    print(note_intercepts)

    print("Group 1 (helpful and not polarizing) avg helpfulness score: {}".format(
        avg(note_intercepts[:n_note // 3])))
    print("Group 2 (helpful and polarizing) avg helpfulness score: {}".format(
        avg(note_intercepts[n_note // 3: -n_note // 3])))
    print("Group 3 (unhelpful) avg helpfulness score: {}".format(
        avg(note_intercepts[n_note - n_note // 3: n_note])))
