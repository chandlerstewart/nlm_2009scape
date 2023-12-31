package content.region.asgarnia.falador.dialogue;

import core.game.component.Component;
import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.node.entity.npc.NPC;
import core.game.node.entity.player.Player;
import core.plugin.Initializable;
import core.game.node.item.Item;

/**
 * Handles the MakeOverMageDialogue dialogue.
 * @author 'Vexia
 */
@Initializable
public class MakeOverMageDialogue extends DialoguePlugin {

	public MakeOverMageDialogue() {

	}

	public MakeOverMageDialogue(Player player) {
		super(player);
	}

	@Override
	public DialoguePlugin newInstance(Player player) {

		return new MakeOverMageDialogue(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		if (args.length == 2) {

			interpreter.sendDialogues(npc, FacialExpression.HALF_ASKING, "Hmm... you didn't feel any unexpected growths", "anywhere around your head just then did you?");
			stage = 600;
			return true;
		}
		interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Hello there! I am known as the make-over mage! I", "have spent many years researching magics that can", "change your physical appearance!");
		stage = 0;
		return true;
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {
		switch (stage) {
		case 0:
			interpreter.sendDialogues(npc, FacialExpression.HAPPY, "I can alter your physical form for a small fee of", "only 3000 gold coins! Would you like me to perform my", "magics upon you?");
			stage = 1;
			break;
		case 1:
			if (!player.getInventory().contains(7803, 1) && !player.getBank().contains(7803, 1) && !player.getEquipment().contains(7803, 1)) {
				interpreter.sendOptions("Select an Option", "Tell me more about this 'make-over'.", "Sure. Do it.", "No thanks.", "Cool amulet! Can I have one?");
			} else {
				interpreter.sendOptions("Select an Option", "Tell me more about this 'make-over'.", "Sure. Do it.", "No thanks.");
			}
			stage = 2;
			break;
		case 2:
			switch (buttonId) {
			case 1:
				interpreter.sendDialogues(player, FacialExpression.HAPPY, "Tell me more about this 'make-over'.");
				stage = 10;
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.HAPPY, "Sure. Do it.");
				stage = 20;
				break;
			case 3:
				interpreter.sendDialogues(player, FacialExpression.HAPPY, "No thanks. I'm happy as Saradomin made me.");
				stage = 19;
				break;
			case 4:
				interpreter.sendDialogues(player, FacialExpression.HAPPY, "Cool amulet! Can I have one?");
				stage = 40;
				break;
			}
			break;
		case 20:
			interpreter.sendDialogues(npc, FacialExpression.HAPPY, "You of course agree that if by some accident you", "are turned into a frog you have no rights for", "compensation or refund.");
			stage = 25;
			break;
		case 21:
			interpreter.sendDialogues(npc, FacialExpression.ANGRY, "Well, go get it then. No freebies here!");
			stage = 22;
			break;
		case 22:
			end();
			break;
		case 25:
			if(!player.getEquipment().isEmpty()){
				interpreter.sendDialogue("You need to take off all your equipment before the mage", "can change your appearance.");
				stage = 900;
			} else {
				end();
				player.getInterfaceManager().open(new Component(205));
			}
			break;
		case 10:
			interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Why, of course! Basically, and I will try and explain", "this so that you will understand it correctly,");
			stage = 11;
			break;
		case 11:
			interpreter.sendDialogues(npc, FacialExpression.FRIENDLY, "I use my secret magical technique to melt your body", "down into a puddle of its elements.");
			stage = 12;
			break;
		case 12:
			interpreter.sendDialogues(npc, FacialExpression.FRIENDLY, "When I have broken down all trace of your body, I", "then rebuild it into the form I am thinking of.");
			stage = 13;
			break;
		case 13:
			interpreter.sendDialogues(npc, FacialExpression.FRIENDLY, "Or, you know, somewhere vaguely close enough", "anyway.");
			stage = 14;
			break;
		case 14:
			interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "Uh... that doesn't sound particularly safe to me...");
			stage = 15;
			break;
		case 15:
			interpreter.sendDialogues(npc, FacialExpression.LAUGH, "It's as safe as houses! Why, I have only had thirty-six", "major accidents this month!");
			stage = 16;
			break;
		case 16:
			interpreter.sendDialogues(npc, FacialExpression.ASKING, "So what do you say? Feel like a change?");
			stage = 17;
			break;
		case 17:
			interpreter.sendOptions("Select an Option", "Sure. Do it.", "No thanks.");
			stage = 18;
			break;
		case 18:
			switch (buttonId) {
			case 1:
				interpreter.sendDialogues(player, FacialExpression.HAPPY, "Sure. Do it.");
				stage = 20;
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.HAPPY, "No thanks. I'm happy as Saradomin made me.");
				stage = 19;
				break;
			}
			break;
		case 19:
			interpreter.sendDialogues(npc, FacialExpression.HALF_GUILTY, "Ehhh... suit yourself.");
			stage = 900;
			break;
		case 900:
			end();
			break;
		case 40:
			interpreter.sendDialogues(npc, FacialExpression.HAPPY, "No problem, but please remember that the amulet I will", "sell you is only a copy of my own. It contains no", "magical powers; and as such will only cost you 100", "coins.");
			stage = 41;
			break;
		case 41:
			interpreter.sendOptions("Select an Option", "Sure, here you go.", "No way! that's too expensive.");
			stage = 42;
			break;
		case 42:
			switch (buttonId) {
			case 1:
				interpreter.sendDialogues(player, FacialExpression.HAPPY, "Sure, here you go.");
				stage = 100;
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.AMAZED, "No way! That's too expensive.");
				stage = 400;
				break;

			}
			break;
		case 400:
			interpreter.sendDialogues(npc, FacialExpression.HALF_GUILTY, "That's fair enough, my jewellery is not to everyone's", "taste.");
			stage = 401;
			break;
		case 401:
			end();
			break;
		case 600:
			interpreter.sendDialogues(player, FacialExpression.EXTREMELY_SHOCKED, "Uh... No...?");
			stage = 601;
			break;
		case 601:
			interpreter.sendDialogues(npc, FacialExpression.WORRIED, "Good, good, I was worried for a second there!");
			stage = 602;
			break;
		case 602:
			interpreter.sendDialogues(player, FacialExpression.SUSPICIOUS, "Uh... Thanks, I guess.");
			stage = 603;
			break;
		case 603:
			end();
			break;
		case 100:
			if (player.getInventory().freeSlots() == 0) {
				end();
				player.getPacketDispatch().sendMessage("You don't have enough inventory space.");
			}
			if (!player.getInventory().contains(995, 100)) {
				end();
				player.getPacketDispatch().sendMessage("You need 100 coins.");
			} else {
				// you recieve an amulet in exchange for 100 coins.
				// 7803
				Item remove = new Item(995, 100);
				if (!player.getInventory().containsItem(remove)) {
					end();
					return true;
				}
				if (player.getInventory().remove(remove)) {
					interpreter.sendItemMessage(7803, "You receive an amulet in exchange for 100 coins.");
					player.getInventory().add(new Item(7803, 1));
					stage = 101;
				}
			}
			break;
		case 101:
			end();
			break;
		}
		return true;
	}

	@Override
	public int[] getIds() {
		return new int[] { 2676, 599 };
	}
}
