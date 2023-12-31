package content.region.karamja.dialogue;

import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.node.entity.npc.NPC;
import core.game.node.entity.player.Player;
import core.game.node.item.GroundItem;
import core.game.node.item.GroundItemManager;
import core.plugin.Initializable;
import core.game.node.item.Item;

/**
 * Represents the dialogue plugin used for the luthas npc.
 * @author 'Vexia
 * @version 1.0
 */
@Initializable
public final class LuthasDialogue extends DialoguePlugin {

	/**
	 * Represents the coins item.
	 */
	private static final Item COINS = new Item(995, 30);

	/**
	 * Constructs a new {@code LuthasDialogue} {@code Object}.
	 */
	public LuthasDialogue() {
		/**
		 * empty.
		 */
	}

	/**
	 * Constructs a new {@code LuthasDialogue} {@code Object}.
	 * @param player the player.
	 */
	public LuthasDialogue(Player player) {
		super(player);
	}

	@Override
	public DialoguePlugin newInstance(Player player) {
		return new LuthasDialogue(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		if (player.getSavedData().getGlobalData().isLuthasTask()) {
			final int current = player.getSavedData().getGlobalData().getKaramjaBananas();
			if (current >= 10) {
				interpreter.sendDialogues(player, FacialExpression.FRIENDLY, "I've filled a crate with bananas.");
				stage = 905;
				return true;
			}
			interpreter.sendDialogues(npc, FacialExpression.HALF_ASKING, "Have you completed your task yet?");
			stage = 900;
			return true;
		}
		interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Hello I'm Luthas, I run the banana plantation here.");
		stage = 0;
		return true;
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {
		switch (stage) {
		case 0:
			interpreter.sendOptions("Select an Option", "Could you offer me employment on your plantation?", "That customs officer is annoying isn't he?");
			stage = 1;
			break;
		case 1:
			switch (buttonId) {
			case 1:
				interpreter.sendDialogues(player, FacialExpression.ASKING, "Could you offer me employment on your plantation?");
				stage = 10;
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.SUSPICIOUS, "That customs officer is annoying isn't she?");
				stage = 20;
				break;
			}
			break;
		case 10:
			interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Yes, I can sort something out. There's a crate ready to", "be loaded onto the ship.");
			stage = 11;
			break;
		case 11:
			interpreter.sendDialogues(npc, FacialExpression.NEUTRAL, "You wouldn't believe the demand for bananas from", "Wydin's shop over in Port Sarim. I think this is the", "third crate I've shipped him this month..");
			stage = 12;
			break;
		case 12:
			interpreter.sendDialogues(npc, FacialExpression.HAPPY, "If you could go fill it up with bananas, I'll pay you 30", "gold.");
			stage = 13;
			player.getSavedData().getGlobalData().setLuthasTask(true);
			break;
		case 13:
			end();
			break;
		case 20:
			interpreter.sendDialogues(npc, FacialExpression.NEUTRAL, "Well I know her pretty well. She doesn't cause me any", "trouble any more");
			stage = 21;
			break;
		case 21:
			interpreter.sendDialogues(npc, FacialExpression.NEUTRAL, "She doesn't even search my export crates anymore.", "She knows they only contain bananas.");
			stage = 22;
			break;
		case 22:
			interpreter.sendDialogues(player, FacialExpression.SUSPICIOUS, "Really? How interesting. Whereabouts do you send", "those to?");
			stage = 23;
			break;
		case 23:
			interpreter.sendDialogues(npc, FacialExpression.NEUTRAL, "There is a little shop over in Port Sarim that buys", "them up by the crate. I believe it is run by a man", "called Wydin.");
			stage = 24;
			break;
		case 24:
			end();
			break;
		case 900:
			int amt = player.getSavedData().getGlobalData().getKaramjaBananas();
			if (amt < 30) {
				interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "No, the crate isn't full yet.");
				stage = 901;
			} else {

			}
			break;
		case 901:
			interpreter.sendDialogues(npc, FacialExpression.NEUTRAL, "Well come back when it is.");
			stage = 21;
			break;
		case 905:
			interpreter.sendDialogues(npc, FacialExpression.HAPPY, "Well done, here's your payment.");
			stage = 906;
			break;
		case 906:
			end();
			player.getPacketDispatch().sendMessage("Luthas hands you 30 coins.");
			player.getSavedData().getGlobalData().setKaramjaBannanas(0);
			player.getSavedData().getGlobalData().setLuthasTask(false);
			if (player.getAttribute("stashed-rum", false)) {
				player.removeAttribute("stashed-rum");
				player.setAttribute("/save:wydin-rum", true);
			}
			if (!player.getInventory().add(COINS)) {
				GroundItemManager.create(new GroundItem(COINS, player.getLocation(), player));
			}
			break;
		}
		return true;
	}

	@Override
	public int[] getIds() {
		return new int[] { 379 };
	}
}
