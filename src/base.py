import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

schools = [
    'abjuration', 'abjuration',
    'necromancy', 'necromancy',
    'evocation', 'evocation',
    'transmutation', 'transmutation',
    'illusion', 'illusion',
    'enchantment', 'enchantment',
    'divination', 'divination',
    'conjuration', 'conjuration'
]

descriptions = [
    "You have resistance to acid, cold, fire, lightning, and thunder damage for the spell’s duration. When you take damage of one of those types, you can use your reaction to gain immunity to that type of damage, including against the triggering damage.",
    "Your spell bolsters your allies with toughness and resolve. Choose up to three creatures within range. Each target's hit point maximum and current hit points increase by 5 for the duration. At Higher Levels.When you cast this spell using a spell slot of 3rd level or higher, a target's hit points increase by an additional 5 for each slot level above 2nd.",
    "You unleash negative energy toward a creature you can see within range. The target makes a Constitution saving throw, taking 7d8 + 30 Necrotic damage on a failed save or half as much damage on a successful one. A Humanoid killed by this spell rises at the start of your next turn as aZombiethat follows your verbal orders.",
    "You touch a dead Humanoid or a piece of one. If the creature has been dead no longer than 10 days, the spell forms a new body for it and calls the soul to enter that body. Roll 1d10 and consult the table below to determine the body’s species, or the DM chooses another playable species. The reincarnated creature makes any choices that a species’ description offers, and the creature recalls its former life. It retains the capabilities it had in its original form, except it loses the traits of its previous species and gains the traits of its new one.",
    "A creature you touch regains a number of hit points equal to 1d8 + your spellcasting ability modifier. This spell has no effect on undead or constructs. At Higher Levels.When you cast this spell using a spell slot of 2nd level or higher, the healing increases by 1d8 for each slot level above 1st.",
    "A loud noise erupts from a point of your choice within range. Each creature in a 10-foot-radiusSpherecentered there makes a Constitution saving throw, taking 3d8 Thunder damage on a failed save or half as much damage on a successful one. A Construct has Disadvantage on the save. A nonmagical object that isn’t being worn or carried also takes the damage if it’s in the spell’s area.",
    "You touch a dead humanoid or a piece of a dead humanoid. Provided that the creature has been dead no longer than 10 days, the spell forms a new adult body for it and then calls the soul to enter that body. If the target's soul isn't free or willing to do so, the spell fails. The magic fashions a new body for the creature to inhabit, which likely causes the creature's race to change. The GM rolls a d100 and consults the following table to determine what form the creature takes when restored to life, or the GM chooses a form.",
    "You gain the ability to move or manipulate creatures or objects by thought. When you cast the spell, and as your action each round for the duration, you can exert your will on one creature or object that you can see within range, causing the appropriate effect below. You can affect the same target round after round, or choose a new one at any time. If you switch targets, the prior target is no longer affected by the spell. Creature.You can try to move a Huge or smaller creature. Make an ability check with your spellcasting ability contested by the creature's Strength check. If you win the contest, you move the creature up to 30 feet in any direction, including upward but not beyond the range of this spell. Until the end of your next turn, the creature isrestrainedin your telekinetic grip. A creature lifted upward is suspended in mid-air.",
    "You create the image of an object, a creature, or some other visible phenomenon that is no larger than a 20-foot cube. The image appears at a spot that you can see within range and lasts for the duration. It seems completely real, including sounds, smells, and temperature appropriate to the thing depicted. You can't create sufficient heat or cold to cause damage, a sound loud enough to deal thunder damage or deafen a creature, or a smell that might sicken a creature (like a troglodyte's stench). As long as you are within range of the illusion, you can use your action to cause the image to move to any other spot within range. As the image changes location, you can alter its appearance so that its movements appear natural for the image. For example, if you create an image of a creature and move it, you can alter the image so that it appears to be walking. Similarly, you can cause the illusion to make different sounds at different times, even making it carry on a conversation, for example.",
    "You implant a message within an object in range—a message that is uttered when a trigger condition is met. Choose an object that you can see and that isn’t being worn or carried by another creature. Then speak the message, which must be 25 words or fewer, though it can be delivered over as long as 10 minutes. Finally, determine the circumstance that will trigger the spell to deliver your message. When that trigger occurs, a magical mouth appears on the object and recites the message in your voice and at the same volume you spoke. If the object you chose has a mouth or something that looks like a mouth (for example, the mouth of a statue), the magical mouth appears there, so the words appear to come from the object’s mouth. When you cast this spell, you can have the spell end after it delivers its message, or it can remain and repeat its message whenever the trigger occurs.",
    "This spell attracts or repels creatures of your choice. You target something within range, either a Huge or smaller object or creature or an area that is no larger than a 200-foot cube. Then specify a kind of intelligent creature, such as red dragons, goblins, or vampires. You invest the target with an aura that either attracts or repels the specified creatures for the duration. Choose antipathy or sympathy as the aura's effect. Antipathy.The enchantment causes creatures of the kind you designated to feel an intense urge to leave the area and avoid the target. When such a creature can see the target or comes within 60 feet of it, the creature must succeed on a Wisdom saving throw or becomefrightened. The creature remainsfrightenedwhile it can see the target or is within 60 feet of it. Whilefrightenedby the target, the creature must use its movement to move to the nearest safe spot from which it can't see the target. If the creature moves more than 60 feet from the target and can't see it, the creature is no longerfrightened, but the creature becomesfrightenedagain if it regains sight of the target or moves within 60 feet of it.",
    "A creature of your choice that you can see within range perceives everything as hilariously funny and falls into fits of laughter if this spell affects it. The target must succeed on a Wisdom saving throw or fallprone, becomingincapacitatedand unable to stand up for the duration. A creature with an Intelligence score of 4 or less isn't affected. At the end of each of its turns, and each time it takes damage, the target can make another Wisdom saving throw. The target has advantage on the saving throw if it's triggered by damage. On a success, the spell ends.",
    "For the duration, you understand the literal meaning of any language that you hear or see signed. You also understand any written language that you see, but you must be touching the surface on which the words are written. It takes about 1 minute to read one page of text. This spell doesn’t decode symbols or secret messages.",
    "For the duration, you sense the location of any Aberration, Celestial, Elemental, Fey, Fiend, or Undead within 30 feet of yourself. You also sense whether theHallowspell is active there and, if so, where. The spell is blocked by 1 foot of stone, dirt, or wood; 1 inch of metal; or a thin sheet of lead.",
    "Until the spell ends, freezing rain and sleet fall in a 20-foot-tall cylinder with a 40-foot radius centered on a point you choose within range. The area is heavily obscured, and exposed flames in the area are doused. The ground in the area is covered with slick ice, making it difficult terrain. When a creature enters the spell's area for the first time on a turn or starts its turn there, it must make a Dexterity saving throw. On a failed save, it fallsprone.",
    "You create a wall of tangled brush bristling with needle-sharp thorns. The wall appears within range on a solid surface and lasts for the duration. You choose to make the wall up to 60 feet long, 10 feet high, and 5 feet thick or a circle that has a 20-foot diameter and is up to 20 feet high and 5 feet thick. The wall blocks line of sight. When the wall appears, each creature in its area makes a Dexterity saving throw, taking 7d8 Piercing damage on a failed save or half as much damage on a successful one."
]

def load():
    data = np.load('data/feiticos_embeddings.npz')

    train_embeddings = data['train_embeddings']
    validation_embeddings = data['validation_embeddings']
    test_embeddings = data['test_embeddings']

    train_label = data['train_label']
    validation_label = data['validation_label']
    test_label = data['test_label']

    scaler = StandardScaler()

    train_embeddings_scaled = scaler.fit_transform(train_embeddings)
    validation_embeddings_scaled = scaler.transform(validation_embeddings)
    test_embeddings_scaled = scaler.transform(test_embeddings)

    return train_label, validation_label, test_label, train_embeddings_scaled, validation_embeddings_scaled, test_embeddings_scaled, scaler

def get_sentence_transformer(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> SentenceTransformer:
    return SentenceTransformer(model_name)

def prediction(school, description, scaler, best_model, transform_model):
    sen_embed = transform_model.encode(description, convert_to_numpy=True)

    sen_embed_scaled = scaler.transform([sen_embed])

    prediction = best_model.predict(sen_embed_scaled)

    if prediction[0] == school:
        print(f"[Acerto] O modelo acertou a escola: {prediction[0]}.")
        return 1
    else:
        print(f"[Erro] A escola correta era {school}, o modelo previu: {prediction[0]}.")
        return 0

def interactive_menu(best_model, scaler, model, info_str: str):
    acertos = 0
    total = 0

    while True:
        print("\n-----OPÇÕES-----")
        print("1 - Testar para um input personalizado.")
        print("2 - Testar para dados pré-feito.")
        print("3 - Sair")
        print("Escolha:", end=" ")
        resp = input()
        print()

        if resp == "1":
            school = input("Digite a escola: ")
            description = input("Digite a descrição do feitiço: ")

            print(info_str)
            acertos += prediction(school, description, scaler, best_model, model)
            total += 1

        elif resp == "2":
            print(info_str)
            for i in range(len(schools)):
                acertos += prediction(schools[i], descriptions[i], scaler, best_model, model)
                total += 1

        elif resp == "3":
            accuracy = acertos / total if total else 0.0
            print(f"Taxa de acerto total: {accuracy:.2%}.")
            print("Saindo.")
            return accuracy

        else:
            print("Erro no input.")
