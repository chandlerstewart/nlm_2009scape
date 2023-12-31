package content.region.misthalin.varrock.dialogue;

import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.global.Skillcape;
import core.game.node.entity.skill.Skills;
import core.game.node.entity.npc.NPC;
import core.plugin.Initializable;
import core.game.node.entity.player.Player;

/**
 * Represents the dialogue plugin used for the head chef.
 * @author 'Vexia
 * @version 1.0
 */
@Initializable
public final class HeadChefDialogue extends DialoguePlugin {

	/**
	 * Constructs a new {@code HeadChefDialogue} {@code Object}.
	 */
	public HeadChefDialogue() {
		/**
		 * empty.
		 */
	}

	/**
	 * Constructs a new {@code HeadChefDialogue} {@code Object}.
	 * @param player the player.
	 */
	public HeadChefDialogue(Player player) {
		super(player);
	}

	@Override
	public DialoguePlugin newInstance(Player player) {
		return new HeadChefDialogue(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		boolean door = false;
		if (args.length == 2)
			door = (boolean) args[1];
		if (door) {
			interpreter.sendDialogues(847, FacialExpression.NEUTRAL, "You can't come in here unless you're wearing a chef's", "hat or something like that.");
			stage = 0;
			return true;
		}
		if (player.getSkills().getStaticLevel(Skills.COOKING) >= 99) {
			interpreter.sendDialogues(847, FacialExpression.HAPPY, "Hello, welcome to the Cooking Guild. It's always great to", "have such an accomplished chef visit. Say would you be", "interested in a Skillcape of Cooking? They're only available", "to master chefs.");
			stage = 100;
			return true;
		}
		interpreter.sendDialogues(847, FacialExpression.HAPPY, "Hello, welcome to the Cooking Guild. Only accomplished", "chefs and cooks are allowed in here. Feel free to use any", "of our facilities.");
		stage = 1;
		return true;
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {

		switch (stage) {
		case 0:
			end();
			break;
		case 1:
			interpreter.sendOptions("Select an Option", "Nice cape you're wearing!", "Thanks, bye.");
			stage = 2;
			break;
		case 2:

			switch (buttonId) {
			case 1:
				interpreter.sendDialogues(player, FacialExpression.HAPPY, "Nice cape, you're wearing!");
				stage = 10;
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.FRIENDLY, "Thanks, bye.");
				stage = 3;
				break;
			}

			break;
		case 3:
			end();
			break;
		case 10:
			interpreter.sendDialogues(847, FacialExpression.HAPPY, "Thank you! It's my most prized possession, it's a Skillcape", "of Cooking; it shows that I've achieved level 99 Cooking", "and am one of the best chefs in the land!");
			stage = 11;
			break;
		case 11:
			end();
			break;
		case 100:
			interpreter.sendOptions("Select an Option", "No thanks.", "Yes please.");
			stage = 101;
			break;
		case 101:

			switch (buttonId) {
			case 1:
				interpreter.sendDialogues(player, FacialExpression.FRIENDLY, "No thanks.");
				stage = 110;
				break;
			case 2:
				player("Yes, please.");
				stage = 150;
				break;
			}
			break;

		case 110:
			interpreter.sendDialogues(847, FacialExpression.HAPPY, "Okay, come back to me if you change your mind.");
			stage = 111;
			break;
		case 111:
			end();
			break;
		case 150:
			interpreter.sendDialogues(847, FacialExpression.HALF_GUILTY, "Most certainly, just as soon as you give me 99000 gold", "coins.");
			stage = 151;
			break;
		case 151:
			interpreter.sendOptions("Select an Option", "That's much too expensive.", "Sure.");
			stage = 152;
			break;
		case 152:

			switch (buttonId) {
			case 1:
				interpreter.sendDialogues(player, FacialExpression.EXTREMELY_SHOCKED, "That's much too expensive.");
				stage = 160;
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.HAPPY, "Sure.");
				stage = 200;
				break;
			}

			break;
		case 160:
			interpreter.sendDialogues(847, FacialExpression.HALF_GUILTY, "I'm sorry you feel that way.");
			stage = 161;
			break;
		case 161:
			end();
			break;
		case 200:
			if (Skillcape.purchase(player, Skills.COOKING)) {
				interpreter.sendDialogues(847, FacialExpression.HAPPY, "Now you can use the title Master Chef.");
			}
			stage = 202;
			break;
		case 202:
			end();
			break;
		}

		return true;
	}

	@Override
	public int[] getIds() {
		return new int[] { 847 };
	}
}
