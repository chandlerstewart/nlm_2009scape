package content.region.karamja.brimhaven.dialogue;

import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.node.entity.npc.NPC;
import core.plugin.Initializable;
import core.game.node.entity.player.Player;

/**
 * Represents the dialogue plugin used for the davon npc.
 * @author 'Vexia
 * @version 1.0
 */
@Initializable
public final class DavonDialogue extends DialoguePlugin {

	/**
	 * Constructs a new {@code DavonDialogue} {@code Object}.
	 */
	public DavonDialogue() {
		/**
		 * empty.
		 */
	}

	/**
	 * Constructs a new {@code DavonDialogue} {@code Object}.
	 * @param player the player.
	 */
	public DavonDialogue(Player player) {
		super(player);
	}

	@Override
	public DialoguePlugin newInstance(Player player) {
		return new DavonDialogue(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		interpreter.sendDialogues(npc, FacialExpression.SUSPICIOUS, "Pssst! Come here if you want to do some amulet", "trading.");
		stage = 0;
		return true;
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {
		switch (stage) {
		case 0:
			interpreter.sendOptions("Select an Option", "What are you selling?", "What do you mean pssst?", "Why don't you ever restock some types of amulets?");
			stage = 1;
			break;
		case 1:
			switch (buttonId) {
			case 1:
				interpreter.sendDialogues(player, FacialExpression.HAPPY, "What are you selling?");
				stage = 10;
				break;
			case 2:
				interpreter.sendDialogues(player, FacialExpression.SUSPICIOUS, "What do you mean pssst?");
				stage = 20;
				break;
			case 3:
				interpreter.sendDialogues(player, FacialExpression.HALF_ASKING, "Why don't you ever restock some types of amulets?");
				stage = 30;
				break;
			}
			break;
		case 10:
			end();
			npc.openShop(player);
			break;
		case 20:
			interpreter.sendDialogues(npc, FacialExpression.SUSPICIOUS, "Errr, I was...I was clearing my throat.");
			stage = 21;
			break;
		case 21:
			end();
			break;
		case 30:
			interpreter.sendDialogues(npc, FacialExpression.HALF_GUILTY, "Some of these amulets are very hard to get. I have to", "wait until an adventurer supplies me.");
			stage = 31;
			break;
		case 31:
			end();
			break;
		}
		return true;
	}

	@Override
	public int[] getIds() {
		return new int[] { 588 };
	}
}
