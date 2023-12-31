package content.region.morytania.canifis.dialogue;

import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.node.entity.npc.NPC;
import core.plugin.Initializable;
import core.game.node.entity.player.Player;

/**
 * Handles the RufusDialogue dialogue.
 * @author 'Vexia
 */
@Initializable
public class RufusDialogue extends DialoguePlugin {

	public RufusDialogue() {

	}

	public RufusDialogue(Player player) {
		super(player);
	}

	@Override
	public DialoguePlugin newInstance(Player player) {

		return new RufusDialogue(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		interpreter.sendDialogues(player, FacialExpression.HAPPY, "Hi!");
		stage = 0;
		return true;
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {
		switch (stage) {
		case 0:
			interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Grreeting frrriend! Welcome to my worrrld famous", "food emporrium! All my meats are so frrresh you'd", "swear you killed them yourrrself!");
			stage = 1;
			break;
		case 1:
			interpreter.sendOptions("Select an Option", "Why do you only sell meats?", "Do you sell cooked food?", "Can I buy some food?");
			stage = 2;
			break;
		case 2:
			switch (buttonId) {
			case 1:
				interpreter.sendDialogues(player, FacialExpression.ASKING, "Why do you only sell meats?");
				stage = 10;
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.ASKING, "Do you sell cooked food?");
				stage = 20;
				break;
			case 3:
				interpreter.sendDialogues(player, FacialExpression.FRIENDLY, "Can I buy some food?");
				stage = 30;
				break;
			}
			break;
		case 10:
			interpreter.sendDialogues(npc, FacialExpression.DISGUSTED, "What? Why, what else would you want to eat? What", "kind of lycanthrrope are you anyway?");
			stage = 11;
			break;
		case 11:
			interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "...A vegetarian one?");
			stage = 12;
			break;
		case 12:
			interpreter.sendDialogues(npc, FacialExpression.EXTREMELY_SHOCKED, "Vegetarrrian...?");
			stage = 13;
			break;
		case 13:
			interpreter.sendDialogues(player, FacialExpression.SUSPICIOUS, "Never mind.");
			stage = 14;
			break;
		case 14:
			end();
			break;
		case 20:
			interpreter.sendDialogues(npc, FacialExpression.EVIL_LAUGH, "Cooked food? Who would want that? You lose all the", "flavourrr of the meat when you can't taste the blood!");
			stage = 21;
			break;
		case 21:
			end();
			break;
		case 30:
			interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Cerrrtainly!");
			stage = 31;
			break;
		case 31:
			end();
			npc.openShop(player);
			break;
		}
		return true;
	}

	@Override
	public int[] getIds() {
		return new int[] { 1038 };
	}
}
