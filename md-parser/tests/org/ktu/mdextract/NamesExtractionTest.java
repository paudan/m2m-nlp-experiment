package org.ktu.mdextract;

import com.nomagic.magicdraw.core.Application;
import com.nomagic.magicdraw.core.Project;
import com.nomagic.magicdraw.openapi.uml.SessionManager;
import com.nomagic.magicdraw.tests.MagicDrawTestCase;
import com.nomagic.magicdraw.uml.symbols.DiagramPresentationElement;
import com.nomagic.uml2.ext.jmi.helpers.ModelHelper;
import com.nomagic.uml2.ext.magicdraw.auxiliaryconstructs.mdmodels.Model;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Package;
import org.junit.Test;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Level;
import java.util.logging.Logger;

public class NamesExtractionTest extends MagicDrawTestCase {

    protected static Project project = null;
    protected SessionManager sessionManager = SessionManager.getInstance();

    @Override
    protected void setUpTest() throws Exception {
        super.setUpTest();
        setSkipMemoryTest(true);
        setMemoryTestReady(false);
        if (sessionManager.isSessionCreated())
            sessionManager.closeSession();
    }

    protected void endTest() {
        if (sessionManager.isSessionCreated())
            sessionManager.cancelSession();
        try {
            Thread.sleep(10000);
        } catch (InterruptedException e) {
            Logger.getLogger(NamesExtractionTest.class.getName()).log(Level.SEVERE, null, e);
        }
    }

    private void loadProjectFile(Path filename) throws IOException {
        if (filename != null && (project == null || !project.isLoaded()))
            project = openProject(filename.normalize().toUri().getPath());
        if (project == null || !project.isLoaded())
            throw new IOException("File " + filename + " was not opened or could not be found!");
        if (project == null)
            project = Application.getInstance().getProject();
        assertNotNull(project);
        sessionManager.createSession("Perform tests");
    }

    @Test
    public void testAtcsFile() {
        try {
            System.out.println("Processing ATCS file");
            loadProjectFile(Paths.get("tests", "resources", "atcs.mdzip"));
            Package root = (Package) ModelHelper.findInParent(project.getPrimaryModel(), "Use Case View", Model.class, true);
            ElementNamesExtractor extractor = new ElementNamesExtractor(root);
            System.out.println(extractor.candidateElements.size());
            extractor.extract("atcs");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testAtcsFileByDiagram() {
        try {
            System.out.println("Processing ATCS file by diagram");
            loadProjectFile(Paths.get("tests", "resources", "atcs.mdzip"));
            Package root = (Package) ModelHelper.findInParent(project.getPrimaryModel(), "Use Case View", Model.class, true);
            ElementNamesExtractor extractor = new ElementNamesExtractor(root);
            extractor.extractByDiagram("atcs");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testBpmnDiagramsFile() {
        try {
            System.out.println("Processing BPMN diagrams file");
            loadProjectFile(Paths.get("tests", "resources", "bpmn diagrams.mdzip"));
            Package root = project.getPrimaryModel();
            ElementNamesExtractor extractor = new BPMNElementNamesExtractor(root);
            System.out.println(extractor.candidateElements.size());
            extractor.extract("bpmn_diagrams");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testBpmnDiagramsFileByDiagram() {
        try {
            System.out.println("Processing BPMN diagrams file (by diagram)");
            loadProjectFile(Paths.get("tests", "resources", "bpmn diagrams.mdzip"));
            Package root = project.getPrimaryModel();
            ElementNamesExtractor extractor = new BPMNElementNamesExtractor(root);
            extractor.extractByDiagram("bpmn_diagrams");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
