package org.ktu.mdextract;

import com.nomagic.magicdraw.cbm.BPMNConstants;
import com.nomagic.magicdraw.cbm.BPMNHelper;
import com.nomagic.magicdraw.cbm.profiles.BPMN2Profile;
import com.nomagic.magicdraw.core.Application;
import com.nomagic.magicdraw.core.Project;
import com.nomagic.magicdraw.uml.symbols.DiagramPresentationElement;
import com.nomagic.uml2.ext.jmi.helpers.StereotypesHelper;
import com.nomagic.uml2.ext.magicdraw.actions.mdbasicactions.Action;
import com.nomagic.uml2.ext.magicdraw.activities.mdfundamentalactivities.Activity;
import com.nomagic.uml2.ext.magicdraw.activities.mdfundamentalactivities.ActivityNode;
import com.nomagic.uml2.ext.magicdraw.activities.mdintermediateactivities.ActivityPartition;
import com.nomagic.uml2.ext.magicdraw.activities.mdstructuredactivities.StructuredActivityNode;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Diagram;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Element;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.NamedElement;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Package;
import com.nomagic.uml2.ext.magicdraw.mdprofiles.Profile;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;


public class BPMNElementNamesExtractor extends ElementNamesExtractor {

    private Project project;
    private Profile bpmnProfile;

    public BPMNElementNamesExtractor(Package root) {
        super(root);
        init();
    }

    public BPMNElementNamesExtractor(Collection<DiagramPresentationElement> diagrams) {
        super(diagrams);
        init();
    }

    private void init() {
        project = Application.getInstance().getProject();
        bpmnProfile = StereotypesHelper.getProfileByURI(project, new File(BPMNConstants.BPMN2_PROFILE_FILENAME).toURI().toString());
    }

    public Collection<DiagramPresentationElement> getDiagrams(Package model, Collection<DiagramPresentationElement> diagrams) {
        Project project = Application.getInstance().getProject();
        if (project == null || model == null)
            return diagrams;
        for (Diagram diag : model.getOwnedDiagram()) {
            DiagramPresentationElement pres = project.getDiagram(diag);
            if (pres != null && BPMNHelper.isBPMNDiagram(pres))
                diagrams.add(pres);
        }
        for (Activity el : BPMNHelper.getBPMNProcesses(model))
            for (Diagram diag : el.getOwnedDiagram()) {
                DiagramPresentationElement pres = project.getDiagram(diag);
                if (pres != null && BPMNHelper.isBPMNDiagram(pres))
                    diagrams.add(pres);
            }
        for (Package pkg : model.getNestedPackage())
            diagrams = getDiagrams(pkg, diagrams);
        return diagrams;
    }

    protected Collection<Element> getDiagramElements(Collection<Element> colelem, DiagramPresentationElement diagram) {
        Collection<Element> newelem = new HashSet<>();
        for (Element element : colelem)
            if (element instanceof Package && diagram.findPresentationElement(element, null) != null)
                addPackageElements(diagram, (Package) element, newelem);
            else if (element instanceof StructuredActivityNode && diagram.findPresentationElement(element, null) != null)
                addSubProcessElements(diagram, (StructuredActivityNode) element, newelem);
        return newelem;
    }

    private void addSubProcessElements(DiagramPresentationElement diagram, StructuredActivityNode node, Collection<Element> elements) {
        for (Element el : node.getOwnedElement())
            if (diagram.findPresentationElement(el, null) != null)
                elements.add(el);
        for (ActivityNode innerNode : node.getNode())
            if (innerNode instanceof StructuredActivityNode && diagram.findPresentationElement(innerNode, null) != null)
                addSubProcessElements(diagram, (StructuredActivityNode) innerNode, elements);
    }

    private Set<String[]> extractTaskRelations(Collection<Element> candidateElements) {
        Set<String[]> containRelations = new HashSet<>();
        Set<Action> tasks = candidateElements.stream().filter(BPMN2Profile::isTask).map(e -> (Action)e).collect(Collectors.toSet());
        for (Action el: tasks) {
            String taskName = el.getName().trim();
            if (taskName.length() == 0) continue;
            Collection<ActivityPartition> parts = el.getInPartition();
            for (ActivityPartition part: parts) {
                NamedElement subject = part.getRepresents() != null ? (NamedElement) part.getRepresents() : part;
                String subjectName = subject.getName().trim();
                if (subjectName.length() == 0)  continue;
                containRelations.add(new String[]{subjectName, taskName});
            }
        }
        return  containRelations;
    }

    public void extract(String prefix) {
        prefix = prefix == null ? "" : (prefix + "_");
        Set<Action> tasks = candidateElements.stream().filter(BPMN2Profile::isTask).map(e -> (Action)e).collect(Collectors.toSet());
        String output = String.join("\n", tasks.stream().map(NamedElement::getName).collect(Collectors.toSet()));
        writeFile(Paths.get("output", prefix + "tasks.csv").toString(), output);
        output = String.join("\n", extractTaskRelations(candidateElements).stream().map(p -> p[0] + ";" + p[1]).collect(Collectors.toSet()));
        writeFile(Paths.get("output", prefix + "lanes_tasks.csv").toString(), output);
    }

    public void extractByDiagram(String prefix) {
        prefix = prefix == null ? "" : (prefix + "_");
        String filename = Paths.get("output", prefix + "lanes_tasks_diag.csv").toString();
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8))) {
            for (Entry<DiagramPresentationElement, Collection<Element>> entry: elementsByDiagram.entrySet()) {
                String diagramName = entry.getKey().getName();
                if (diagramName == null) continue;
                Set<String[]> relations = extractTaskRelations(entry.getValue());
                String output = String.join("\n", relations.stream().map(p -> diagramName + ";" + p[0] + ";" + p[1]).collect(Collectors.toSet()));
                writer.write(output);
                writer.newLine();
            }
        } catch (IOException e) {
            Logger.getLogger(ElementNamesExtractor.class.getName()).log(Level.SEVERE, null, e);
        }
    }
}
