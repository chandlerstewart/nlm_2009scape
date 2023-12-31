package content.region.misthalin.barbvillage.stronghold.playersafety;

import core.game.component.Component;
import core.game.dialogue.DialoguePlugin;
import core.game.dialogue.FacialExpression;
import core.game.node.entity.npc.NPC;
import core.game.node.entity.player.Player;
import core.tools.Log;
import core.tools.SystemLogger;
import core.game.system.task.Pulse;
import core.game.world.GameWorld;

import static core.api.ContentAPIKt.log;

/**
 * @author Tyler Telis
 */
public class GuardDialoguePlugin extends DialoguePlugin {

	/**
	 * Constructs a new {@code GuardDialogue} instance.
	 * @param player The {@code Player} instance.
	 */
	public GuardDialoguePlugin(Player player) {
		super(player);
	}

	/**
	 * Constructs a new {@code GuardDialoguePlugin} instance.
	 */
	public GuardDialoguePlugin() {
	}

	@Override
	public DialoguePlugin newInstance(Player player) {
		return new GuardDialoguePlugin(player);
	}

	@Override
	public boolean open(Object... args) {
		npc = (NPC) args[0];
		boolean read = player.getSavedData().getGlobalData().hasReadPlaques();
		interpreter.sendDialogues(npc, read ? FacialExpression.HALF_GUILTY : FacialExpression.FURIOUS, read ? "Can I help you?" : "Ahem! Can I help you?");
		stage = 0;
		return true;
	}

	@Override
	public boolean handle(int interfaceId, int buttonId) {
		boolean read = player.getSavedData().getGlobalData().hasReadPlaques();
		switch (stage) {
		case -1:// END dialogue stage.
			end();
			break;
		case 0:
			interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, read ? "Can I go upstairs?" : "I'd like to go up to the training centre please.");
			increment();
			break;
		case 1:
			interpreter.sendDialogues(npc, FacialExpression.FURIOUS, !read ? "Sorry, citizen, you can't go up there." : "", "Yes, citizen. Before you do I am instructed to give", "you one final piece of information");
			increment();
			break;
		case 2:
			interpreter.sendDialogues(player, read ? FacialExpression.HALF_GUILTY : FacialExpression.OLD_SNEAKY, read ? "Oh, okay then." : "Why not?");
			increment();
			break;
		case 3:
			if (!read) {
				npc("You must learn about player safety before<br>entering the training centre.");
			} else {
				npc("In your travels around " + GameWorld.getSettings().getName() + ", should you find a", "player who acts in a way that breaks on of our rules,", "you should report them.");
			}
			increment();
			break;
		case 4:
			interpreter.sendDialogues(read ? npc : player, FacialExpression.HALF_GUILTY, !read ? "Oh. How do I do that?" : "Reporting is very simple and easy to do. Simply click", "the Report Abuse button at the bottom of the screen", "and you will be shown the following screen:");
			stage = read ? 10 : stage + 1;
			break;
		case 5:
			interpreter.sendDialogues(npc, FacialExpression.HALF_GUILTY, "Each of these gublinches have been caught breaking the", "Rules of " + GameWorld.getSettings().getName() + ". You should read the plaques on", "each of their cells to learn what they did wrong.");
			increment();
			break;
		case 6:
			interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "Oh, right. I can enter the training centre once I have", "done that?");
			increment();
			break;
		case 7:
			interpreter.sendDialogues(npc, FacialExpression.HALF_GUILTY, "Yes. Once you have have examined each of the plaques,", "come and speak to me and I will tell you about the", "Report Abuse function.");
			increment();
			break;
		case 8:
			interpreter.sendDialogues(npc, FacialExpression.HALF_GUILTY, "After that, I can let you into the training centre,", "upstairs.");
			increment();
			break;
		case 9:
			interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "Okay, thanks for the help.");
			player.getPacketDispatch().sendMessage("You need to read the jail plaques before the guard will allow you upstairs.");
			stage = -1;
			break;
		case 10:
			if (read) {
				player.getInterfaceManager().open(new Component(700));
				GameWorld.getPulser().submit(new Pulse(5) {

					@Override
					public boolean pulse() {
						if (player != null) {
							player.getInterfaceManager().close();
						}
						return true;
					}

				});
				stage = 12;
			}
			break;
		case 12:
			interpreter.sendDialogues(npc, FacialExpression.HALF_GUILTY, "Simply enter the player's name in the box and click the", "rule that the offender was breaking.");
			stage = 13;
			break;
		case 13:
			interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "Okay. Then what?");
			stage = 14;
			break;
		case 14:
			interpreter.sendDialogues(npc, FacialExpression.HALF_GUILTY, "That's it! It reall is that simple and it only takes a ", "moment to do. Now you may enter the training", "centre. Good luck, citizen.");
			stage = 15;
			break;
		case 15:
			interpreter.sendDialogues(player, FacialExpression.HALF_GUILTY, "Thanks!");
			stage = -1;
			break;
		default:
			log(this.getClass(), Log.ERR,  "Unhandled dialogue stage=" + stage);
		}
		return false;
	}

	@Override
	public int[] getIds() {
		return new int[] { 7142 };
	}

}
